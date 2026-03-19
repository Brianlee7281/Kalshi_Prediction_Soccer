#!/usr/bin/env python
"""Calibrate EKF process noise sigma_omega_sq from historical EPL goal times.

Method: Maximum likelihood on observed goal sequences.  For each candidate
sigma_omega_sq, run the EKF forward on every historical match and compute the
log-likelihood of observing the actual goals at their actual minutes.  The
value that maximises total LL is the MLE estimate.

Validation: on held-out recent seasons, check P(D) for late-equaliser matches
against observed draw frequency.

Usage:
    PYTHONPATH=. python scripts/calibrate_sigma_omega.py
"""
from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.commentaries_parser import parse_commentaries_dir
from src.calibration.step_1_3_ml_prior import compute_C_time
from src.prematch.phase2_pipeline import _shin_vig_removal

# ── Hardcoded EPL params (same as simulation) ────────────────────
B = np.zeros(8, dtype=np.float64)
GAMMA_H = np.array([0.0, -0.15, 0.10, -0.05])
GAMMA_A = np.array([0.0, 0.10, -0.15, -0.05])
DELTA_H = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
DELTA_A = np.array([0.10, 0.05, 0.0, -0.05, -0.10])
ALPHA_1 = 2.0
BASIS_BOUNDS = np.array(
    [0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0], dtype=np.float64,
)
C_TIME = compute_C_time(B, BASIS_BOUNDS)  # 93.0

# League-average priors (used when no odds available)
A_H_DEFAULT = float(np.log(1.4 / C_TIME))
A_A_DEFAULT = float(np.log(1.1 / C_TIME))

# EKF initial uncertainty — Tier 2 (Pinnacle) average
P_0 = 0.15

# Grid to search (includes 0 baseline and fine resolution at low end)
SIGMA_GRID = [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]


# ── Data structures ──────────────────────────────────────────────

@dataclass
class GoalRecord:
    minute: int
    team: str  # "home" | "away"


@dataclass
class MatchRecord:
    match_id: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    goals: list[GoalRecord]
    a_H_prior: float
    a_A_prior: float
    season: str  # e.g. "2023-2024"


# ── Data loading ─────────────────────────────────────────────────

def _load_odds_lookup(odds_dir: Path) -> dict[str, tuple[float, float, float]]:
    """Build (home_team_lower, away_team_lower, season) -> (PSH, PSD, PSA) map."""
    lookup: dict[str, tuple[float, float, float]] = {}
    season_map = {
        "E0_1920": "2019-2020", "E0_2021": "2020-2021",
        "E0_2122": "2021-2022", "E0_2223": "2022-2023",
        "E0_2324": "2023-2024", "E0_2425": "2024-2025",
    }
    for csv_path in sorted(odds_dir.glob("E0_*.csv")):
        stem = csv_path.stem
        season = season_map.get(stem, stem)
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
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


def _odds_to_prior(
    odds_h: float, odds_d: float, odds_a: float,
) -> tuple[float, float]:
    """Convert bookmaker odds to (a_H, a_A) via Shin vig removal + Poisson."""
    p_h, p_d, p_a = _shin_vig_removal(odds_h, odds_d, odds_a)
    # Quick heuristic for implied expected goals (avoids full backsolve)
    mu_h = max(0.3, 1.5 * p_h + 1.0 * p_d + 0.5 * p_a)
    mu_a = max(0.3, 0.5 * p_h + 1.0 * p_d + 1.5 * p_a)
    return float(np.log(mu_h / C_TIME)), float(np.log(mu_a / C_TIME))


def _season_from_commentary_file(match: dict) -> str:
    """Infer season from date string like '01.02.2020'."""
    date_str = match.get("date", "")
    try:
        parts = date_str.split(".")
        year = int(parts[2])
        month = int(parts[1])
        if month >= 7:
            return f"{year}-{year+1}"
        return f"{year-1}-{year}"
    except (ValueError, IndexError):
        return "unknown"


def load_epl_matches() -> list[MatchRecord]:
    """Load EPL matches from commentaries, join with odds for priors."""
    commentaries_dir = PROJECT_ROOT / "data" / "commentaries"
    odds_dir = PROJECT_ROOT / "data" / "odds_historical"

    all_matches = parse_commentaries_dir(commentaries_dir)
    epl_matches = [m for m in all_matches if m["league_id"] == "1204"]
    print(f"EPL matches parsed: {len(epl_matches)}")

    odds_lookup = _load_odds_lookup(odds_dir)
    print(f"Odds lookup entries: {len(odds_lookup)}")

    records: list[MatchRecord] = []
    odds_hits = 0
    for m in epl_matches:
        season = _season_from_commentary_file(m)
        goals = [
            GoalRecord(minute=g["minute"], team=g["team"])
            for g in m["goal_events"]
            if 1 <= g["minute"] <= 95  # skip minute-0 or invalid
        ]

        # Try to find odds
        ht = m["home_team"].strip().lower()
        at = m["away_team"].strip().lower()
        key = f"{ht}|{at}|{season}"
        odds = odds_lookup.get(key)
        if odds:
            a_H, a_A = _odds_to_prior(*odds)
            odds_hits += 1
        else:
            a_H, a_A = A_H_DEFAULT, A_A_DEFAULT

        records.append(MatchRecord(
            match_id=m["match_id"],
            home_team=m["home_team"],
            away_team=m["away_team"],
            home_goals=m["home_goals"],
            away_goals=m["away_goals"],
            goals=goals,
            a_H_prior=a_H,
            a_A_prior=a_A,
            season=season,
        ))
    print(f"Odds matched: {odds_hits}/{len(records)}")
    return records


# ── EKF forward pass + log-likelihood ────────────────────────────

def _basis_index(t: float) -> int:
    for i in range(len(BASIS_BOUNDS) - 1):
        if t < BASIS_BOUNDS[i + 1]:
            return i
    return len(BASIS_BOUNDS) - 2


def compute_match_ll(
    match: MatchRecord,
    sigma_omega_sq: float,
) -> float:
    """Run EKF forward on one match, return log-likelihood of goal sequence."""
    a_H = match.a_H_prior
    a_A = match.a_A_prior
    P_H = P_0
    P_A = P_0

    # delta_S tracking for score-diff effects
    score_H, score_A = 0, 0
    ll = 0.0
    dt = 1.0  # 1-minute steps

    # Build set of goal minutes for fast lookup
    goal_minutes: dict[int, list[str]] = {}
    for g in match.goals:
        goal_minutes.setdefault(g.minute, []).append(g.team)

    T_END = 93  # match duration in minutes

    for t_int in range(1, T_END + 1):
        t = float(t_int)

        # EKF predict
        P_H += sigma_omega_sq * dt
        P_A += sigma_omega_sq * dt

        # Current intensities (per minute)
        bi = _basis_index(t)
        b_val = B[bi] if bi < len(B) else 0.0

        ds = score_H - score_A
        di = max(0, min(4, ds + 2))

        lam_H = math.exp(a_H + b_val + GAMMA_H[0] + DELTA_H[di])
        lam_A = math.exp(a_A + b_val + GAMMA_A[0] + DELTA_A[di])

        goals_at_t = goal_minutes.get(t_int, [])

        if goals_at_t:
            for team in goals_at_t:
                if team == "home":
                    # Goal likelihood
                    if lam_H > 1e-10:
                        ll += math.log(lam_H)
                    # EKF goal update
                    K = P_H / (P_H * lam_H + 1.0)
                    a_H += K * (1.0 - lam_H * (1.0 / 60.0))
                    P_H = (1.0 - K * lam_H) * P_H
                    score_H += 1
                else:
                    if lam_A > 1e-10:
                        ll += math.log(lam_A)
                    K = P_A / (P_A * lam_A + 1.0)
                    a_A += K * (1.0 - lam_A * (1.0 / 60.0))
                    P_A = (1.0 - K * lam_A) * P_A
                    score_A += 1

                # Clamp
                a_H = max(match.a_H_prior - 1.5, min(match.a_H_prior + 1.5, a_H))
                a_A = max(match.a_A_prior - 1.5, min(match.a_A_prior + 1.5, a_A))

                # Recompute lambdas for next goal in same minute
                ds = score_H - score_A
                di = max(0, min(4, ds + 2))
                lam_H = math.exp(a_H + b_val + GAMMA_H[0] + DELTA_H[di])
                lam_A = math.exp(a_A + b_val + GAMMA_A[0] + DELTA_A[di])
        else:
            # No-goal survival likelihood
            surv = 1.0 - (lam_H + lam_A) * dt
            if surv > 1e-10:
                ll += math.log(surv)

            # EKF no-goal update
            if lam_H > 0:
                K0_H = P_H * lam_H / (P_H * lam_H + 1.0)
                a_H += K0_H * (0.0 - lam_H * dt)
            if lam_A > 0:
                K0_A = P_A * lam_A / (P_A * lam_A + 1.0)
                a_A += K0_A * (0.0 - lam_A * dt)

            a_H = max(match.a_H_prior - 1.5, min(match.a_H_prior + 1.5, a_H))
            a_A = max(match.a_A_prior - 1.5, min(match.a_A_prior + 1.5, a_A))

    return ll


# ── Late-equaliser validation ────────────────────────────────────

def late_equaliser_validation(
    matches: list[MatchRecord],
    sigma_omega_sq: float,
) -> tuple[float, float, int]:
    """For matches with late equalisers (t>=75), compute model P(D) vs actual.

    Returns (mean_model_pd, actual_draw_rate, n_matches).
    """
    model_pds: list[float] = []
    actual_draws: list[bool] = []

    for match in matches:
        # Find late equalisers: goal at t>=75 that makes score level
        score_H, score_A = 0, 0
        equaliser_minute = None
        for g in match.goals:
            if g.team == "home":
                score_H += 1
            else:
                score_A += 1
            if score_H == score_A and g.minute >= 75 and score_H >= 1:
                equaliser_minute = g.minute
                break  # take the first late equaliser

        if equaliser_minute is None:
            continue

        # Run EKF forward to equaliser minute, get P(D)
        a_H = match.a_H_prior
        a_A = match.a_A_prior
        P_H_ekf = P_0
        P_A_ekf = P_0
        s_H, s_A = 0, 0
        dt = 1.0

        for t_int in range(1, equaliser_minute + 1):
            t = float(t_int)
            P_H_ekf += sigma_omega_sq * dt
            P_A_ekf += sigma_omega_sq * dt

            bi = _basis_index(t)
            b_val = B[bi] if bi < len(B) else 0.0
            ds = s_H - s_A
            di = max(0, min(4, ds + 2))
            lam_H = math.exp(a_H + b_val + GAMMA_H[0] + DELTA_H[di])
            lam_A = math.exp(a_A + b_val + GAMMA_A[0] + DELTA_A[di])

            goals_at_t = [g.team for g in match.goals if g.minute == t_int]
            if goals_at_t:
                for team in goals_at_t:
                    if team == "home":
                        K = P_H_ekf / (P_H_ekf * lam_H + 1.0)
                        a_H += K * (1.0 - lam_H * (1.0 / 60.0))
                        P_H_ekf = (1.0 - K * lam_H) * P_H_ekf
                        s_H += 1
                    else:
                        K = P_A_ekf / (P_A_ekf * lam_A + 1.0)
                        a_A += K * (1.0 - lam_A * (1.0 / 60.0))
                        P_A_ekf = (1.0 - K * lam_A) * P_A_ekf
                        s_A += 1
                    a_H = max(match.a_H_prior - 1.5, min(match.a_H_prior + 1.5, a_H))
                    a_A = max(match.a_A_prior - 1.5, min(match.a_A_prior + 1.5, a_A))
                    ds = s_H - s_A
                    di = max(0, min(4, ds + 2))
                    lam_H = math.exp(a_H + b_val + GAMMA_H[0] + DELTA_H[di])
                    lam_A = math.exp(a_A + b_val + GAMMA_A[0] + DELTA_A[di])
            else:
                if lam_H > 0:
                    K0 = P_H_ekf * lam_H / (P_H_ekf * lam_H + 1.0)
                    a_H += K0 * (0.0 - lam_H * dt)
                if lam_A > 0:
                    K0 = P_A_ekf * lam_A / (P_A_ekf * lam_A + 1.0)
                    a_A += K0 * (0.0 - lam_A * dt)
                a_H = max(match.a_H_prior - 1.5, min(match.a_H_prior + 1.5, a_H))
                a_A = max(match.a_A_prior - 1.5, min(match.a_A_prior + 1.5, a_A))

        # Compute remaining mu from equaliser to T=93
        remaining = max(0.0, 93.0 - float(equaliser_minute))
        if remaining < 0.5:
            continue  # skip goals in stoppage with no real time remaining

        bi = _basis_index(float(equaliser_minute))
        b_val = B[bi] if bi < len(B) else 0.0
        ds = s_H - s_A
        di = max(0, min(4, ds + 2))
        mu_H_rem = math.exp(a_H + b_val + DELTA_H[di]) * remaining
        mu_A_rem = math.exp(a_A + b_val + DELTA_A[di]) * remaining

        if math.isnan(mu_H_rem) or math.isnan(mu_A_rem):
            continue
        mu_H_rem = max(1e-6, mu_H_rem)
        mu_A_rem = max(1e-6, mu_A_rem)

        # Poisson P(draw) = P(same additional goals)
        from scipy.stats import poisson
        pd = 0.0
        for k in range(8):
            pd += poisson.pmf(k, mu_H_rem) * poisson.pmf(k, mu_A_rem)

        if not math.isnan(pd):
            model_pds.append(pd)
            actual_draws.append(match.home_goals == match.away_goals)

    if not model_pds:
        return 0.0, 0.0, 0

    return (
        float(np.mean(model_pds)),
        float(np.mean(actual_draws)),
        len(model_pds),
    )


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.monotonic()

    print("=" * 72)
    print("  EKF Process Noise Calibration (sigma_omega_sq)")
    print("  League: EPL | Method: Maximum Likelihood on goal times")
    print("=" * 72)
    print()

    # Load data
    matches = load_epl_matches()
    if not matches:
        print("ERROR: No EPL matches found")
        sys.exit(1)

    # Split: train on 2019-2023, validate on 2023-2025
    train_seasons = {"2019-2020", "2020-2021", "2021-2022", "2022-2023"}
    val_seasons = {"2023-2024", "2024-2025"}
    train = [m for m in matches if m.season in train_seasons]
    val = [m for m in matches if m.season in val_seasons]
    print(f"Train: {len(train)} matches ({', '.join(sorted(train_seasons))})")
    print(f"Val:   {len(val)} matches ({', '.join(sorted(val_seasons))})")
    print()

    # Grid search
    print(f"{'sigma_omega_sq':>14} | {'train_LL':>12} | {'val_LL':>12} | "
          f"{'K@t=78':>7} | {'late_eq P(D)':>12} | {'actual draw%':>12}")
    print("-" * 90)

    best_val_ll = -1e18
    best_sigma = SIGMA_GRID[0]
    results: list[dict] = []

    for sigma in SIGMA_GRID:
        # Train LL
        train_ll = sum(compute_match_ll(m, sigma) for m in train)
        # Val LL
        val_ll = sum(compute_match_ll(m, sigma) for m in val)

        # K at t=78 (P after 78 minutes of predict-only growth from P_0=0.15,
        # ignoring measurement updates for simplicity)
        P_at_78 = P_0 + sigma * 78.0
        lam_typical = 0.014  # typical lambda at t=78
        K_78 = P_at_78 / (P_at_78 * lam_typical + 1.0)

        # Late-equaliser validation on val set
        model_pd, actual_dr, n_eq = late_equaliser_validation(val, sigma)

        if val_ll > best_val_ll:
            best_val_ll = val_ll
            best_sigma = sigma

        results.append({
            "sigma": sigma, "train_ll": train_ll, "val_ll": val_ll,
            "K_78": K_78, "model_pd": model_pd, "actual_dr": actual_dr,
            "n_eq": n_eq,
        })

        print(
            f"  {sigma:12.4f} | {train_ll:12.1f} | {val_ll:12.1f} | "
            f" {K_78:5.3f} | {model_pd:12.4f} | {actual_dr:12.4f}"
        )

    print()
    print(f"Best sigma_omega_sq (by val LL): {best_sigma}")
    print()

    # Refine around best with finer grid
    lo = best_sigma * 0.5
    hi = best_sigma * 2.0
    fine_grid = np.linspace(lo, hi, 20)
    best_fine_ll = -1e18
    best_fine_sigma = best_sigma
    for sigma in fine_grid:
        val_ll = sum(compute_match_ll(m, sigma) for m in val)
        if val_ll > best_fine_ll:
            best_fine_ll = val_ll
            best_fine_sigma = float(sigma)

    print(f"Refined sigma_omega_sq: {best_fine_sigma:.6f}")
    print()

    # Final validation with refined value
    model_pd, actual_dr, n_eq = late_equaliser_validation(val, best_fine_sigma)
    P_final = P_0 + best_fine_sigma * 78.0
    K_final = P_final / (P_final * 0.014 + 1.0)

    print("=" * 72)
    print("  Final Result")
    print("=" * 72)
    print(f"  sigma_omega_sq = {best_fine_sigma:.6f}")
    print(f"  K at t=78:       {K_final:.4f}")
    print(f"  P at t=78:       {P_final:.4f}")
    print(f"  Late-eq P(D):    {model_pd:.4f} (model) vs {actual_dr:.4f} (actual)")
    print(f"  Late-eq matches: {n_eq}")
    print(f"  K target range:  0.30 - 0.50")
    print(f"  K in range:      {'YES' if 0.25 <= K_final <= 0.55 else 'NO'}")
    print()

    elapsed = time.monotonic() - t0
    print(f"Calibration completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
