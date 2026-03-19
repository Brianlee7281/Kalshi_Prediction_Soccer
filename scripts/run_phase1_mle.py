#!/usr/bin/env python
"""Phase 1 MLE calibration: estimate b, delta, gamma from EPL historical data.

Usage:
    PYTHONPATH=. python scripts/run_phase1_mle.py
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.commentaries_parser import parse_commentaries_dir
from src.calibration.phase1_mle import (
    MatchData, Phase1MLEResult, precompute_batch, run_phase1_mle, total_ll_vec,
)
from src.calibration.step_1_3_ml_prior import compute_C_time
from src.prematch.phase2_pipeline import _shin_vig_removal

# ── Constants ─────────────────────────────────────────────────────
ALPHA_1 = 2.0
BASIS_BOUNDS = np.array(
    [0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0], dtype=np.float64,
)
WIDTHS = np.diff(BASIS_BOUNDS)  # [15, 15, 17, 15, 15, 10, 5, 1]
C_TIME = float(np.sum(WIDTHS))  # 93 with flat b


# ── Data loading ──────────────────────────────────────────────────

def _season_from_date(date_str: str) -> str:
    try:
        parts = date_str.split(".")
        year = int(parts[2])
        month = int(parts[1])
        return f"{year}-{year+1}" if month >= 7 else f"{year-1}-{year}"
    except (ValueError, IndexError):
        return "unknown"


def _load_odds_lookup(odds_dir: Path) -> dict[str, tuple[float, float, float]]:
    lookup: dict[str, tuple[float, float, float]] = {}
    season_map = {
        "E0_1920": "2019-2020", "E0_2021": "2020-2021",
        "E0_2122": "2021-2022", "E0_2223": "2022-2023",
        "E0_2324": "2023-2024", "E0_2425": "2024-2025",
    }
    for csv_path in sorted(odds_dir.glob("E0_*.csv")):
        season = season_map.get(csv_path.stem, csv_path.stem)
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    ht = (row.get("HomeTeam") or "").strip().lower()
                    at = (row.get("AwayTeam") or "").strip().lower()
                    try:
                        psh = float(row.get("PSH") or row.get("PSCH") or "0")
                        psd = float(row.get("PSD") or row.get("PSCD") or "0")
                        psa = float(row.get("PSA") or row.get("PSCA") or "0")
                    except ValueError:
                        continue
                    if psh > 0 and psd > 0 and psa > 0:
                        lookup[f"{ht}|{at}|{season}"] = (psh, psd, psa)
        except Exception:
            continue
    return lookup


def _odds_to_prior(h: float, d: float, a: float) -> tuple[float, float]:
    p_h, p_d, p_a = _shin_vig_removal(h, d, a)
    mu_h = max(0.3, 1.5 * p_h + 1.0 * p_d + 0.5 * p_a)
    mu_a = max(0.3, 0.5 * p_h + 1.0 * p_d + 1.5 * p_a)
    return float(np.log(mu_h / C_TIME)), float(np.log(mu_a / C_TIME))


def load_epl_match_data() -> tuple[list[MatchData], list[MatchData], dict]:
    """Load EPL matches, split into train/val, return (train, val, stats)."""
    commentaries_dir = PROJECT_ROOT / "data" / "commentaries"
    odds_dir = PROJECT_ROOT / "data" / "odds_historical"

    all_matches = parse_commentaries_dir(commentaries_dir)
    epl = [m for m in all_matches if m["league_id"] == "1204"]
    print(f"EPL matches parsed: {len(epl)}")

    odds_lookup = _load_odds_lookup(odds_dir)
    print(f"Odds lookup entries: {len(odds_lookup)}")

    a_H_default = float(np.log(1.4 / C_TIME))
    a_A_default = float(np.log(1.1 / C_TIME))

    train_seasons = {"2019-2020", "2020-2021", "2021-2022", "2022-2023"}
    val_seasons = {"2023-2024", "2024-2025"}

    train: list[MatchData] = []
    val: list[MatchData] = []
    odds_hits = 0
    total_goals = 0

    for m in epl:
        season = _season_from_date(m.get("date", ""))
        if season not in train_seasons and season not in val_seasons:
            continue

        goals_h = [g["minute"] + 0.5 for g in m["goal_events"]
                    if g["team"] == "home" and 1 <= g["minute"] <= 95]
        goals_a = [g["minute"] + 0.5 for g in m["goal_events"]
                    if g["team"] == "away" and 1 <= g["minute"] <= 95]
        total_goals += len(goals_h) + len(goals_a)

        ht = m["home_team"].strip().lower()
        at = m["away_team"].strip().lower()
        key = f"{ht}|{at}|{season}"
        odds = odds_lookup.get(key)
        if odds:
            a_H, a_A = _odds_to_prior(*odds)
            odds_hits += 1
        else:
            a_H, a_A = a_H_default, a_A_default

        md = MatchData(a_H=a_H, a_A=a_A,
                       goal_times_home=goals_h, goal_times_away=goals_a)

        if season in train_seasons:
            train.append(md)
        else:
            val.append(md)

    print(f"Odds matched: {odds_hits}/{len(train)+len(val)}")
    print(f"Train: {len(train)}, Val: {len(val)}, Total goals: {total_goals}")

    return train, val, {"total_goals": total_goals, "odds_hits": odds_hits}


# ── Validation tables ─────────────────────────────────────────────

def print_goals_per_period(matches: list[MatchData], b: np.ndarray, label: str) -> None:
    """Compare empirical vs model goal rates per basis period."""
    empirical_goals = np.zeros(8)
    empirical_exposure = np.zeros(8)

    for m in matches:
        for t in m.goal_times_home + m.goal_times_away:
            bi = 0
            for i in range(len(BASIS_BOUNDS) - 1):
                if t < BASIS_BOUNDS[i + 1]:
                    bi = i
                    break
            else:
                bi = 7
            empirical_goals[bi] += 1
        for i in range(8):
            empirical_exposure[i] += WIDTHS[i]

    empirical_rate = empirical_goals / empirical_exposure
    flat_rate = empirical_goals.sum() / empirical_exposure.sum()

    print(f"\n  Goals per period ({label}, {len(matches)} matches):")
    print(f"  {'Period':>6} {'Minutes':>10} {'Goals':>7} {'Emp rate':>10} {'Flat rate':>10} {'Model rate':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(8):
        lo = BASIS_BOUNDS[i]
        hi = BASIS_BOUNDS[i + 1]
        # Model rate = exp(b[i]) * flat_rate (normalized)
        model_rate = flat_rate * math.exp(b[i])
        print(f"  {i:>6} {lo:5.0f}-{hi:3.0f} {empirical_goals[i]:7.0f} "
              f"{empirical_rate[i]:10.4f} {flat_rate:10.4f} {model_rate:10.4f}")


def print_score_state_table(
    matches: list[MatchData], b: np.ndarray,
    delta_H: np.ndarray, delta_A: np.ndarray, label: str,
) -> None:
    """Compare empirical vs model intensity by score-state."""
    # Accumulate exposure and goals per delta_S bin
    bins = 5  # delta_S in {-2, -1, 0, +1, +2}
    exposure = np.zeros(bins)
    home_goals = np.zeros(bins)

    for m in matches:
        all_g = sorted(
            [(t, "home") for t in m.goal_times_home] +
            [(t, "away") for t in m.goal_times_away],
            key=lambda x: x[0],
        )
        s_H, s_A = 0, 0
        seg_starts = [0.0] + [g[0] for g in all_g] + [m.T]
        gi = 0
        for si in range(len(seg_starts) - 1):
            lo = seg_starts[si]
            hi = seg_starts[si + 1]
            ds = s_H - s_A
            di = max(0, min(4, ds + 2))
            exposure[di] += hi - lo
            if gi < len(all_g) and all_g[gi][0] == hi:
                if all_g[gi][1] == "home":
                    home_goals[di] += 1
                    s_H += 1
                else:
                    s_A += 1
                gi += 1

    emp_rate = np.where(exposure > 0, home_goals / exposure, 0)
    ref_rate = emp_rate[2] if emp_rate[2] > 0 else 0.01

    print(f"\n  Score-state home intensity ({label}):")
    print(f"  {'dS':>4} {'Segments':>10} {'Exposure':>10} {'H goals':>8} "
          f"{'Emp lam_H':>10} {'delta_H':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    ds_labels = ["<=-2", "  -1", "   0", "  +1", " >=+2"]
    for i in range(5):
        print(f"  {ds_labels[i]:>4} {'':>10} {exposure[i]:10.0f} {home_goals[i]:8.0f} "
              f"{emp_rate[i]:10.5f} {delta_H[i]:8.4f}")


def print_home_advantage(
    matches: list[MatchData], b: np.ndarray,
    gamma_H: float, gamma_A: float,
    delta_H: np.ndarray, delta_A: np.ndarray,
    label: str,
) -> None:
    """Home advantage summary."""
    total_h = sum(len(m.goal_times_home) for m in matches)
    total_a = sum(len(m.goal_times_away) for m in matches)
    n = len(matches)
    hw = sum(1 for m in matches if len(m.goal_times_home) > len(m.goal_times_away))

    print(f"\n  Home advantage ({label}, {n} matches):")
    print(f"    Home goals/match: {total_h/n:.2f}")
    print(f"    Away goals/match: {total_a/n:.2f}")
    print(f"    Home win %:       {100*hw/n:.1f}%")
    print(f"    gamma_H: {gamma_H:+.4f}  (exp={math.exp(gamma_H):.3f}x)")
    print(f"    gamma_A: {gamma_A:+.4f}  (exp={math.exp(gamma_A):.3f}x)")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.monotonic()

    print("=" * 72)
    print("  Phase 1 MLE Calibration")
    print("  League: EPL | Method: Sequential L-BFGS on Poisson LL")
    print("=" * 72)
    print()

    train, val, stats = load_epl_match_data()
    if not train:
        print("ERROR: No training data")
        sys.exit(1)

    # Precompute batches for baseline
    train_batch = precompute_batch(train, BASIS_BOUNDS)
    val_batch = precompute_batch(val, BASIS_BOUNDS)

    # Baseline LL with flat params
    b0 = np.zeros(8)
    dH0 = np.zeros(5)
    dA0 = np.zeros(5)
    base_train = total_ll_vec(train_batch, b0, 0.0, 0.0, dH0, dA0)
    base_val = total_ll_vec(val_batch, b0, 0.0, 0.0, dH0, dA0)
    print(f"\nBaseline (flat params): train_LL={base_train:.1f}  val_LL={base_val:.1f}")
    print()

    # Run sequential MLE
    print("Running sequential MLE estimation...")
    result = run_phase1_mle(
        train, val, BASIS_BOUNDS,
        gamma_H_init=0.0, gamma_A_init=0.0,
        max_rounds=3, convergence_tol=0.005,
    )

    # ── Print final parameters ────────────────────────────────
    print()
    print("=" * 72)
    print("  Final Calibrated Parameters")
    print("=" * 72)
    print(f"  b       = {np.round(result.b, 4).tolist()}")
    print(f"  delta_H = {np.round(result.delta_H, 4).tolist()}")
    print(f"  delta_A = {np.round(result.delta_A, 4).tolist()}")
    print(f"  gamma_H = {result.gamma_H:.4f}")
    print(f"  gamma_A = {result.gamma_A:.4f}")
    print(f"  train_LL = {result.train_LL:.1f}  (delta from baseline: {result.train_LL - base_train:+.1f})")
    print(f"  val_LL   = {result.val_LL:.1f}  (delta from baseline: {result.val_LL - base_val:+.1f})")
    print(f"  Rounds: {result.rounds_completed}")

    # ── Validation tables ─────────────────────────────────────
    print()
    print("=" * 72)
    print("  Validation")
    print("=" * 72)

    print_goals_per_period(train + val, result.b, "all EPL")
    print_score_state_table(train + val, result.b, result.delta_H, result.delta_A, "all EPL")
    print_home_advantage(train + val, result.b, result.gamma_H, result.gamma_A,
                         result.delta_H, result.delta_A, "all EPL")

    # ── Print as Python dict for simulate_match.py ────────────
    print()
    print("=" * 72)
    print("  Copy-paste for simulate_match.py HARDCODED_EPL_PARAMS:")
    print("=" * 72)
    print(f'    "b": {np.round(result.b, 6).tolist()},')
    print(f'    "gamma_H": [0.0, -0.15, 0.10, -0.05],  # red-card gamma unchanged')
    print(f'    "gamma_A": [0.0, 0.10, -0.15, -0.05],  # red-card gamma unchanged')
    print(f'    "delta_H": {np.round(result.delta_H, 6).tolist()},')
    print(f'    "delta_A": {np.round(result.delta_A, 6).tolist()},')
    print(f'    # gamma home/away advantage (used as additive to a_H/a_A in intensity):')
    print(f'    # gamma_H_advantage = {result.gamma_H:.6f}')
    print(f'    # gamma_A_advantage = {result.gamma_A:.6f}')

    elapsed = time.monotonic() - t0
    print(f"\nCalibration completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
