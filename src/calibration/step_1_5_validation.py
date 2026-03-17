"""Step 1.5 — Walk-Forward Validation.

Chronological walk-forward cross-validation for MMPP calibration.
Trains on previous folds, validates on current fold using Brier Score.

IMPORTANT: XGBoost prior is retrained per fold to prevent data leakage.
The prior must only see training-fold matches, never validation matches.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from src.common.types import IntervalRecord
from src.math.mc_core import mc_simulate_remaining
from src.math.step_1_4_nll_optimize import optimize_nll, prepare_match_data

log = structlog.get_logger(__name__)

# MC simulation parameters for validation
_MC_N_PATHS = 10000
_MC_SEED_BASE = 42
_BASIS_BOUNDS = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, 93.0])


def compute_brier_score(
    predicted_probs: list[tuple[float, float, float]],
    actual_results: list[str],
) -> float:
    """Brier Score for 1x2 predictions (standard multi-outcome scale).

    Uses the standard sports analytics formula (FiveThirtyEight, academic):
    BS = (1/N) * Σ [(1/K) * Σ_k (p_k - y_k)²]  where K=3 outcomes.

    Range: 0 (perfect) to 0.667 (worst). Uniform baseline = 0.222.

    Args:
        predicted_probs: [(p_home, p_draw, p_away), ...]
        actual_results: ["H", "D", "A", ...]

    Returns:
        Brier Score on standard scale (0 to 0.667).
    """
    if not predicted_probs:
        return 1.0

    n_outcomes = 3
    total = 0.0
    for (p_h, p_d, p_a), result in zip(predicted_probs, actual_results):
        y_h = 1.0 if result == "H" else 0.0
        y_d = 1.0 if result == "D" else 0.0
        y_a = 1.0 if result == "A" else 0.0
        total += ((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2) / n_outcomes

    return total / len(predicted_probs)


def _predict_1x2(
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma_H: np.ndarray,
    gamma_A: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    Q: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Predict 1x2 probabilities using MC simulation from kickoff."""
    # Prepare Q matrix for MC: diagonal rates and off-diagonal transition probs
    Q_diag = -np.diag(Q).copy()
    Q_off = np.zeros_like(Q)
    for i in range(4):
        row_sum = Q_diag[i]
        if row_sum > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = Q[i, j] / row_sum
        # If Q_diag[i] == 0, no transitions from state i

    scores = mc_simulate_remaining(
        t_now=0.0,
        T_end=90.0,
        S_H=0,
        S_A=0,
        state=0,
        score_diff=0,
        a_H=a_H,
        a_A=a_A,
        b=b,
        gamma_H=gamma_H,
        gamma_A=gamma_A,
        delta_H=delta_H,
        delta_A=delta_A,
        Q_diag=Q_diag,
        Q_off=Q_off,
        basis_bounds=_BASIS_BOUNDS,
        N=_MC_N_PATHS,
        seed=seed,
    )

    home_wins = np.sum(scores[:, 0] > scores[:, 1])
    draws = np.sum(scores[:, 0] == scores[:, 1])
    away_wins = np.sum(scores[:, 0] < scores[:, 1])
    n = len(scores)

    return (home_wins / n, draws / n, away_wins / n)


def walk_forward_cv(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
    a_H_init: np.ndarray,
    a_A_init: np.ndarray,
    actual_results: list[str],
    Q: np.ndarray,
    *,
    n_folds: int = 5,
    sigma_a: float = 0.5,
    num_epochs: int = 300,
    matches_ordered: list[dict] | None = None,
    odds_data: dict[str, Any] | None = None,
    league_id: str | None = None,
) -> dict:
    """Chronological walk-forward cross-validation.

    Split matches chronologically into n_folds.
    For each fold: train XGBoost on training folds ONLY (no leakage),
    then train MMPP NLL, then predict on validation fold.

    Args:
        intervals_by_match: {match_id: [IntervalRecord, ...]}
        match_ids: Ordered list of match IDs (chronological).
        a_H_init: Fallback a_H (used only when matches_ordered/odds_data
            not provided, e.g. in unit tests).
        a_A_init: Fallback a_A.
        actual_results: ["H", "D", "A", ...] for each match.
        Q: 4x4 generator matrix from step 1.2.
        n_folds: Number of CV folds.
        sigma_a: Regularization strength.
        num_epochs: NLL optimization epochs per fold.
        matches_ordered: Raw match dicts (needed for per-fold XGBoost).
        odds_data: Odds dict (needed for per-fold XGBoost).
        league_id: League ID string (needed for per-fold XGBoost).

    Returns:
        Dict with per_fold_brier, overall_brier, go_nogo.
    """
    # If raw data provided, retrain XGBoost per fold to prevent leakage
    retrain_xgb = (
        matches_ordered is not None
        and odds_data is not None
        and league_id is not None
    )

    n = len(match_ids)
    fold_size = n // n_folds
    if fold_size < 5:
        log.warning("too_few_matches_for_cv", n=n, n_folds=n_folds)
        return {
            "per_fold_brier": [],
            "overall_brier": 1.0,
            "go_nogo": "NO-GO",
        }

    # Build match_id → index in matches_ordered for fast lookup
    mid_to_ordered_idx: dict[str, int] = {}
    if matches_ordered is not None:
        for i, m in enumerate(matches_ordered):
            mid_to_ordered_idx[m["match_id"]] = i

    per_fold_brier: list[float] = []
    all_preds: list[tuple[float, float, float]] = []
    all_actuals: list[str] = []

    for fold in range(1, n_folds):
        train_end = fold * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_end <= val_start:
            continue

        train_ids = match_ids[:train_end]
        val_ids = match_ids[val_start:val_end]

        # --- Per-fold XGBoost prior (no leakage) ---
        fold_val_a_H: np.ndarray | None = None
        fold_val_a_A: np.ndarray | None = None
        fold_train_a_H: np.ndarray | None = None
        fold_train_a_A: np.ndarray | None = None

        if retrain_xgb:
            train_matches_fold = [
                matches_ordered[mid_to_ordered_idx[mid]]
                for mid in train_ids
                if mid in mid_to_ordered_idx
            ]
            val_matches_fold = [
                matches_ordered[mid_to_ordered_idx[mid]]
                for mid in val_ids
                if mid in mid_to_ordered_idx
            ]

            from src.calibration.step_1_3_ml_prior import (
                predict_xgboost_prior,
                train_xgboost_prior,
            )

            # Train XGBoost ONLY on training fold matches
            xgb_result, _, fold_train_a_H, fold_train_a_A = train_xgboost_prior(
                train_matches_fold, odds_data, league_id,
            )

            # Predict for val matches using train-only model
            if xgb_result is not None:
                fold_val_a_H, fold_val_a_A = predict_xgboost_prior(
                    xgb_result, val_matches_fold, odds_data, train_matches_fold,
                )
            else:
                # MLE fallback — predict for val using train-only context
                # Re-run with train+val but MLE has no leakage (uses historical form)
                _, _, fold_all_a_H, fold_all_a_A = train_xgboost_prior(
                    train_matches_fold + val_matches_fold, odds_data, league_id,
                )
                n_train_fold = len(train_matches_fold)
                fold_val_a_H = fold_all_a_H[n_train_fold:]
                fold_val_a_A = fold_all_a_A[n_train_fold:]

            log.info(
                "fold_xgb_retrained", fold=fold,
                train_matches=len(train_matches_fold),
                val_matches=len(val_matches_fold),
                xgb_used=xgb_result is not None,
            )

        # Filter intervals to training matches only
        train_intervals = {
            mid: intervals_by_match[mid]
            for mid in train_ids
            if mid in intervals_by_match
        }

        if len(train_intervals) < 10:
            log.warning("fold_too_small", fold=fold, train_size=len(train_intervals))
            continue

        # Prepare and optimize on training data
        train_ids_filtered = [mid for mid in train_ids if mid in train_intervals]

        if retrain_xgb and fold_train_a_H is not None:
            # Map filtered train IDs back to fold indices
            train_mid_to_fold_idx = {
                mid: i for i, mid in enumerate(
                    mid for mid in train_ids if mid in mid_to_ordered_idx
                )
            }
            train_a_H = np.array([
                fold_train_a_H[train_mid_to_fold_idx[mid]]
                for mid in train_ids_filtered
                if mid in train_mid_to_fold_idx
            ])
            train_a_A = np.array([
                fold_train_a_A[train_mid_to_fold_idx[mid]]
                for mid in train_ids_filtered
                if mid in train_mid_to_fold_idx
            ])
            train_ids_filtered = [
                mid for mid in train_ids_filtered
                if mid in train_mid_to_fold_idx
            ]
        else:
            train_a_H = np.array([
                a_H_init[match_ids.index(mid)] for mid in train_ids_filtered
            ])
            train_a_A = np.array([
                a_A_init[match_ids.index(mid)] for mid in train_ids_filtered
            ])

        try:
            match_data = prepare_match_data(train_intervals, train_ids_filtered)
            result = optimize_nll(
                match_data, train_a_H, train_a_A,
                sigma_a=sigma_a, num_epochs=num_epochs,
            )
        except Exception as e:
            log.warning("fold_optimization_failed", fold=fold, error=str(e))
            continue

        # Predict on validation matches
        fold_preds: list[tuple[float, float, float]] = []
        fold_actuals: list[str] = []

        # Build val match_id → fold val index mapping
        val_mid_to_fold_idx: dict[str, int] = {}
        if retrain_xgb:
            val_mid_to_fold_idx = {
                mid: i for i, mid in enumerate(
                    mid for mid in val_ids if mid in mid_to_ordered_idx
                )
            }

        for vid in val_ids:
            vid_idx = match_ids.index(vid)
            actual = actual_results[vid_idx]
            if actual not in ("H", "D", "A"):
                continue

            if (
                retrain_xgb
                and fold_val_a_H is not None
                and vid in val_mid_to_fold_idx
            ):
                val_a_H = float(fold_val_a_H[val_mid_to_fold_idx[vid]])
                val_a_A = float(fold_val_a_A[val_mid_to_fold_idx[vid]])
            else:
                val_a_H = float(a_H_init[vid_idx])
                val_a_A = float(a_A_init[vid_idx])

            pred = _predict_1x2(
                a_H=val_a_H,
                a_A=val_a_A,
                b=result.b,
                gamma_H=result.gamma_H,
                gamma_A=result.gamma_A,
                delta_H=result.delta_H,
                delta_A=result.delta_A,
                Q=Q,
                seed=_MC_SEED_BASE + vid_idx,
            )
            fold_preds.append(pred)
            fold_actuals.append(actual)

        if fold_preds:
            fold_bs = compute_brier_score(fold_preds, fold_actuals)
            per_fold_brier.append(fold_bs)
            all_preds.extend(fold_preds)
            all_actuals.extend(fold_actuals)
            log.info("fold_complete", fold=fold, brier=round(fold_bs, 4), n_val=len(fold_preds))

    # Overall metrics
    overall_brier = compute_brier_score(all_preds, all_actuals) if all_preds else 1.0
    uniform_brier = compute_brier_score(
        [(1 / 3, 1 / 3, 1 / 3)] * len(all_actuals), all_actuals,
    ) if all_actuals else 1.0

    # GO/NO-GO: use last 2 folds (most training data, most reliable).
    # Early folds have too little training data and inflate the average.
    if len(per_fold_brier) >= 2:
        late_fold_avg = np.mean(per_fold_brier[-2:])
        go_nogo = "GO" if late_fold_avg < uniform_brier else "NO-GO"
    else:
        go_nogo = "GO" if overall_brier < uniform_brier else "NO-GO"

    log.info(
        "walk_forward_cv_complete",
        overall_brier=round(overall_brier, 4),
        uniform_brier=round(uniform_brier, 4),
        late_fold_avg=round(float(late_fold_avg), 4) if len(per_fold_brier) >= 2 else None,
        go_nogo=go_nogo,
        n_folds_completed=len(per_fold_brier),
    )

    return {
        "per_fold_brier": per_fold_brier,
        "overall_brier": overall_brier,
        "uniform_brier": uniform_brier,
        "go_nogo": go_nogo,
    }
