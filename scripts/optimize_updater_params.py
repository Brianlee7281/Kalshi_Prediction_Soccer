#!/usr/bin/env python3
"""
Optimize correction_cap and sigma_a_sq for InPlayStrengthUpdater
================================================================
Grid search + scipy refinement using Brier Score minimization
on 11,500+ historical matches from data/commentaries/.

Approach:
  1. Parse all matches with goal times and final results
  2. Time-based 80/20 split (train/val)
  3. Grid search over (correction_cap, sigma_a_sq)
  4. For each pair, replay every match goal-by-goal:
     - Use league-average a_H/a_A as prior
     - After each goal, update a_H/a_A via Bayesian shrinkage
     - Evaluate P(result) at 6 time points via Poisson convolution
     - Compute Brier Score vs actual outcome
  5. Refine best grid point with scipy.optimize.minimize

Usage:
    PYTHONPATH=. python scripts/optimize_updater_params.py
"""

from __future__ import annotations

import math
import sys
import time as time_mod
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from src.calibration.commentaries_parser import parse_commentaries_dir

# --- Constants ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
COMMENTARIES_DIR = PROJECT_ROOT / "data" / "commentaries"

# EPL-average MMPP parameters (8-period basis, from Phase 1 calibration)
B = [-0.1688, -0.1282, -0.1008, 0.0581, -0.0025, -0.0797, -0.0499, 0.1982]
GAMMA_H = [0.0, -0.30, 0.10, -0.20]
GAMMA_A = [0.0, 0.10, -0.30, -0.20]
DELTA_H = [0.15, 0.08, 0.0, -0.05, -0.12]
DELTA_A = [-0.12, -0.05, 0.0, 0.08, 0.15]
A_H_AVG = -4.09  # exp(-4.09)*93 ~ 1.55 goals/match
A_A_AVG = -4.33  # exp(-4.33)*93 ~ 1.22 goals/match
T_EXP = 93.0
BASIS_BOUNDS = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, T_EXP]

# Evaluate predictions at these match minutes
EVAL_MINUTES = [15, 30, 45, 60, 75, 85]

# Poisson convolution: max remaining goals per team (covers >99.9% mass)
MAX_REM = 8

# Grid search values
CAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, None]
SIGMAS = [0.01, 0.05, 0.10, 0.25, 0.50]


# --- Core math ---------------------------------------------------------------

def compute_mu_remaining(
    t_min: float, a_H: float, a_A: float, score_diff: int,
) -> tuple[float, float]:
    """Remaining expected goals from t_min to T_EXP.

    Uses piecewise intensity over 8 basis periods.
    Assumes state_X=0 (no red cards) for simplicity.
    """
    ds = max(-2, min(2, score_diff))
    di = ds + 2
    mu_H = 0.0
    mu_A = 0.0
    for k in range(8):
        seg_start = max(t_min, BASIS_BOUNDS[k])
        seg_end = min(T_EXP, BASIS_BOUNDS[k + 1])
        if seg_start >= seg_end:
            continue
        dt = seg_end - seg_start
        mu_H += dt * math.exp(a_H + B[k] + GAMMA_H[0] + DELTA_H[di])
        mu_A += dt * math.exp(a_A + B[k] + GAMMA_A[0] + DELTA_A[di])
    return max(1e-6, mu_H), max(1e-6, mu_A)


# Precompute log-factorial for Poisson PMF
_LOG_FACT = gammaln(np.arange(MAX_REM + 2, dtype=np.float64) + 1)


def outcome_probs(
    mu_H: float, mu_A: float, score_h: int, score_a: int,
) -> tuple[float, float, float]:
    """P(home_win), P(draw), P(away_win) via analytical Poisson convolution.

    Enumerates remaining goals 0..MAX_REM for each team.
    """
    ks = np.arange(MAX_REM + 1, dtype=np.float64)
    log_mu_H = math.log(max(mu_H, 1e-10))
    log_mu_A = math.log(max(mu_A, 1e-10))

    h_pmf = np.exp(ks * log_mu_H - mu_H - _LOG_FACT[:MAX_REM + 1])
    a_pmf = np.exp(ks * log_mu_A - mu_A - _LOG_FACT[:MAX_REM + 1])

    joint = np.outer(h_pmf, a_pmf)

    fh = (score_h + ks)[:, None]
    fa = (score_a + ks)[None, :]

    p_home = float(np.sum(joint[fh > fa]))
    p_draw = float(np.sum(joint[fh == fa]))
    p_away = float(np.sum(joint[fh < fa]))

    return p_home, p_draw, p_away


def bayesian_update(
    a_prior: float, n_actual: int, mu_elapsed: float,
    sigma_a_sq: float, cap: float | None,
) -> float:
    """Replicate InPlayStrengthUpdater._bayesian_update with variable cap."""
    if mu_elapsed <= 0.0:
        return a_prior
    shrink = mu_elapsed / (mu_elapsed + sigma_a_sq)
    correction = math.log((n_actual + 0.5) / (mu_elapsed + 0.5))
    if cap is not None:
        correction = max(-cap, min(cap, correction))
    return a_prior + shrink * correction


# --- Match evaluation --------------------------------------------------------

def evaluate_match(
    goals: list[dict],
    actual_h: float,
    actual_d: float,
    actual_a: float,
    sigma_a_sq: float | None,
    cap: float | None,
) -> tuple[float, int]:
    """Replay one match, return (sum_brier, n_eval_points).

    If sigma_a_sq is None, no strength updates (baseline).
    Evaluates at each EVAL_MINUTE, processing goals chronologically.
    """
    a_H = A_H_AVG
    a_A = A_A_AVG

    # mu_at_kickoff with initial (average) strengths
    mu_H_kick, mu_A_kick = compute_mu_remaining(0.0, a_H, a_A, 0)

    n_H = 0
    n_A = 0
    score_h = 0
    score_a = 0
    goal_idx = 0

    total_bs = 0.0
    n_evals = 0

    for eval_min in EVAL_MINUTES:
        # Process all goals that occurred at or before this eval minute
        while goal_idx < len(goals) and goals[goal_idx]["minute"] <= eval_min:
            g = goals[goal_idx]
            gmin = float(g["minute"])

            if sigma_a_sq is not None:
                # Compute mu_elapsed BEFORE updating score (matches production)
                mu_H_rem, mu_A_rem = compute_mu_remaining(
                    gmin, a_H, a_A, score_h - score_a,
                )
                mu_H_elapsed = max(0.0, mu_H_kick - mu_H_rem)
                mu_A_elapsed = max(0.0, mu_A_kick - mu_A_rem)

            # Update score
            if g["team"] == "home":
                score_h += 1
                n_H += 1
            else:
                score_a += 1
                n_A += 1

            # Update strengths (always recomputes from a_init, not incremental)
            if sigma_a_sq is not None:
                a_H = bayesian_update(A_H_AVG, n_H, mu_H_elapsed, sigma_a_sq, cap)
                a_A = bayesian_update(A_A_AVG, n_A, mu_A_elapsed, sigma_a_sq, cap)

            goal_idx += 1

        # Evaluate prediction at this minute
        mu_H_rem, mu_A_rem = compute_mu_remaining(
            float(eval_min), a_H, a_A, score_h - score_a,
        )
        p_h, p_d, p_a = outcome_probs(mu_H_rem, mu_A_rem, score_h, score_a)

        bs = (p_h - actual_h) ** 2 + (p_d - actual_d) ** 2 + (p_a - actual_a) ** 2
        total_bs += bs
        n_evals += 1

    return total_bs, n_evals


def evaluate_dataset(
    matches: list[dict],
    sigma_a_sq: float | None,
    cap: float | None,
) -> float:
    """Mean Brier Score across all matches and eval points."""
    total_bs = 0.0
    total_evals = 0
    for m in matches:
        bs, ne = evaluate_match(
            m["_goals"], m["_y_h"], m["_y_d"], m["_y_a"],
            sigma_a_sq, cap,
        )
        total_bs += bs
        total_evals += ne
    return total_bs / total_evals if total_evals > 0 else 999.0


# --- Main --------------------------------------------------------------------

def main() -> None:
    print("=" * 76)
    print("OPTIMIZE UPDATER PARAMS: correction_cap + sigma_a_sq")
    print("=" * 76)

    # ── Step 1: Parse and prepare data ──────────────────────────────────────
    print("\n[Step 1] Parsing commentaries data...")
    t0 = time_mod.time()
    raw_matches = parse_commentaries_dir(COMMENTARIES_DIR)
    print(f"  Parsed {len(raw_matches)} matches in {time_mod.time() - t0:.1f}s")

    # Prepare: sort goals, determine result, parse date
    matches: list[dict] = []
    skipped_date = 0
    for m in raw_matches:
        goals = [g for g in m["goal_events"] if g["minute"] > 0]
        # goals already sorted by parser (line 132 of commentaries_parser.py)

        hg, ag = m["home_goals"], m["away_goals"]
        y_h = 1.0 if hg > ag else 0.0
        y_d = 1.0 if hg == ag else 0.0
        y_a = 1.0 if hg < ag else 0.0

        try:
            dt = datetime.strptime(m["date"], "%d.%m.%Y")
        except ValueError:
            skipped_date += 1
            continue

        matches.append({
            "_goals": goals,
            "_y_h": y_h,
            "_y_d": y_d,
            "_y_a": y_a,
            "_date": dt,
            "_n_goals": len(goals),
            "league_id": m["league_id"],
        })

    matches.sort(key=lambda x: x["_date"])
    n_with_goals = sum(1 for m in matches if m["_n_goals"] > 0)

    print(f"  Valid matches: {len(matches)} (skipped {skipped_date} bad dates)")
    print(f"  Date range: {matches[0]['_date']:%Y-%m-%d} to {matches[-1]['_date']:%Y-%m-%d}")
    print(f"  Matches with goals: {n_with_goals} ({100 * n_with_goals / len(matches):.0f}%)")

    # Result distribution
    n_home = sum(1 for m in matches if m["_y_h"] == 1.0)
    n_draw = sum(1 for m in matches if m["_y_d"] == 1.0)
    n_away = sum(1 for m in matches if m["_y_a"] == 1.0)
    print(f"  Results: H={n_home} ({100 * n_home / len(matches):.0f}%) "
          f"D={n_draw} ({100 * n_draw / len(matches):.0f}%) "
          f"A={n_away} ({100 * n_away / len(matches):.0f}%)")

    # Time-based 80/20 split
    split_idx = int(len(matches) * 0.80)
    train = matches[:split_idx]
    val = matches[split_idx:]
    print(f"\n  Train: {len(train)} matches (to {train[-1]['_date']:%Y-%m-%d})")
    print(f"  Val:   {len(val)} matches (from {val[0]['_date']:%Y-%m-%d})")

    # ── Step 2: Baseline (no updater) ────────────────────────────────────────
    print("\n[Step 2] Computing baseline (no updater, fixed a_H/a_A)...")
    t0 = time_mod.time()
    baseline_train = evaluate_dataset(train, sigma_a_sq=None, cap=None)
    baseline_val = evaluate_dataset(val, sigma_a_sq=None, cap=None)
    print(f"  Baseline train Brier: {baseline_train:.6f}")
    print(f"  Baseline val Brier:   {baseline_val:.6f}")
    print(f"  ({time_mod.time() - t0:.1f}s)")

    # ── Step 3: Grid search ──────────────────────────────────────────────────
    total_pts = len(CAPS) * len(SIGMAS)
    print(f"\n[Step 3] Grid search: {len(CAPS)} caps x {len(SIGMAS)} sigmas = {total_pts} points")

    results: list[dict] = []
    t0 = time_mod.time()
    done = 0

    for cap in CAPS:
        for sigma in SIGMAS:
            train_bs = evaluate_dataset(train, sigma_a_sq=sigma, cap=cap)
            val_bs = evaluate_dataset(val, sigma_a_sq=sigma, cap=cap)
            results.append({
                "cap": cap,
                "sigma_a_sq": sigma,
                "train_brier": train_bs,
                "val_brier": val_bs,
                "train_improv": (baseline_train - train_bs) / baseline_train * 100,
                "val_improv": (baseline_val - val_bs) / baseline_val * 100,
            })
            done += 1
            elapsed = time_mod.time() - t0
            eta = elapsed / done * (total_pts - done) if done > 0 else 0
            cap_str = f"{cap:.2f}" if cap is not None else "None"
            sys.stdout.write(
                f"\r  [{done}/{total_pts}] cap={cap_str:>5} sig={sigma:.2f}"
                f" | train={train_bs:.6f} val={val_bs:.6f}"
                f" | ETA {eta:.0f}s   "
            )
            sys.stdout.flush()

    print(f"\n  Grid search completed in {time_mod.time() - t0:.1f}s")

    # Sort by val_brier
    results.sort(key=lambda r: r["val_brier"])

    # Print results table
    print("\n" + "=" * 76)
    print("GRID SEARCH RESULTS (sorted by val_brier)")
    print("=" * 76)

    print(f"\n  {'cap':>5} {'sigma_sq':>9} {'train_brier':>12} "
          f"{'val_brier':>12} {'train_impr':>11} {'val_impr':>10}")
    print("  " + "-" * 67)

    # Baseline row
    print(f"  {'---':>5} {'(none)':>9} {baseline_train:>12.6f} "
          f"{baseline_val:>12.6f} {'---':>11} {'---':>10}  <-- baseline")

    for i, r in enumerate(results):
        cap_str = f"{r['cap']:.2f}" if r["cap"] is not None else " None"
        marker = "  <-- BEST" if i == 0 else ""
        print(
            f"  {cap_str:>5} {r['sigma_a_sq']:>9.2f}"
            f" {r['train_brier']:>12.6f} {r['val_brier']:>12.6f}"
            f" {r['train_improv']:>+10.2f}% {r['val_improv']:>+9.2f}%{marker}"
        )

    best = results[0]
    best_cap = best["cap"]
    best_sigma = best["sigma_a_sq"]
    print(f"\n  BEST GRID POINT: cap={best_cap}, sigma_a_sq={best_sigma}")
    print(f"    Val Brier: {best['val_brier']:.6f} ({best['val_improv']:+.2f}% vs baseline)")

    # ── Step 4: Scipy refinement ─────────────────────────────────────────────
    print(f"\n[Step 4] Refining with scipy.optimize.minimize (Nelder-Mead)...")
    t0 = time_mod.time()

    if best_cap is None:
        # No-cap case: optimize sigma only
        def objective_1d(params: np.ndarray) -> float:
            sigma_val = max(0.001, float(params[0]))
            return evaluate_dataset(val, sigma_a_sq=sigma_val, cap=None)

        result_opt = minimize(
            objective_1d,
            x0=[best_sigma],
            method="Nelder-Mead",
            options={"maxiter": 40, "xatol": 0.005, "fatol": 1e-7},
        )
        opt_cap: float | None = None
        opt_sigma = max(0.001, float(result_opt.x[0]))
        opt_val_brier = float(result_opt.fun)
    else:
        # Optimize both cap and sigma
        def objective_2d(params: np.ndarray) -> float:
            cap_val = max(0.01, float(params[0]))
            sigma_val = max(0.001, float(params[1]))
            return evaluate_dataset(val, sigma_a_sq=sigma_val, cap=cap_val)

        result_opt = minimize(
            objective_2d,
            x0=[best_cap, best_sigma],
            method="Nelder-Mead",
            options={"maxiter": 60, "xatol": 0.005, "fatol": 1e-7},
        )
        opt_cap = max(0.01, float(result_opt.x[0]))
        opt_sigma = max(0.001, float(result_opt.x[1]))
        opt_val_brier = float(result_opt.fun)

    opt_train_brier = evaluate_dataset(train, sigma_a_sq=opt_sigma, cap=opt_cap)

    print(f"  Refinement completed in {time_mod.time() - t0:.1f}s")
    print(f"  Optimizer converged: {result_opt.success} ({result_opt.nit} iterations)")
    print(f"  Refined cap: {opt_cap}")
    print(f"  Refined sigma_a_sq: {opt_sigma:.4f}")
    print(f"  Refined val Brier: {opt_val_brier:.6f}")

    # ── Step 5: Final recommendation ─────────────────────────────────────────
    print("\n" + "=" * 76)
    print("FINAL RECOMMENDATION")
    print("=" * 76)

    opt_val_improv = (baseline_val - opt_val_brier) / baseline_val * 100
    opt_train_improv = (baseline_train - opt_train_brier) / baseline_train * 100
    overfit_gap = abs(opt_train_brier - opt_val_brier)

    cap_display = f"{opt_cap:.4f}" if opt_cap is not None else "None (uncapped)"
    print(f"""
  OPTIMAL correction_cap = {cap_display}
  OPTIMAL sigma_a_sq     = {opt_sigma:.4f}

  Baseline val Brier:  {baseline_val:.6f}
  Optimal val Brier:   {opt_val_brier:.6f}
  Brier improvement vs no-updater baseline: {opt_val_improv:+.2f}%

  Train Brier:         {opt_train_brier:.6f} ({opt_train_improv:+.2f}%)
  Val Brier:           {opt_val_brier:.6f} ({opt_val_improv:+.2f}%)
  Overfitting check:   train-val gap = {overfit_gap:.6f}""")

    if opt_val_improv > 0:
        print("  --> Strength updater IMPROVES predictions")
    elif opt_val_improv > -0.5:
        print("  --> Strength updater is NEUTRAL (within noise)")
    else:
        print("  --> Strength updater HURTS predictions on this dataset")
        print("      (may still help on live data where market prices differ)")

    # Context for interpretation
    print(f"""
  NOTES:
    - Prior uses league-average a_H/a_A for all matches (no Phase 2 backsolve)
    - Updater corrects this generic prior using observed goals
    - With match-specific priors (Phase 2), smaller corrections are expected
    - {len(train) + len(val)} matches across 8 leagues, {len(EVAL_MINUTES)} eval points each
    - Brier evaluated at minutes: {EVAL_MINUTES}
""")


if __name__ == "__main__":
    main()
