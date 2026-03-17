"""Phase 1 Worker — Full Calibration Pipeline.

Orchestrates the entire Phase 1 pipeline for a single league:
parse commentaries → load odds → segment intervals → Q matrix →
XGBoost prior → NLL optimize → validate → DB save.
"""
from __future__ import annotations

import asyncio
import io
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import asyncpg
import numpy as np
import structlog

from src.calibration.commentaries_parser import parse_commentaries_dir
from src.calibration.odds_loader import load_odds_csv
from src.calibration.step_1_1_intervals import segment_all_matches
from src.calibration.step_1_3_ml_prior import train_xgboost_prior
from src.calibration.step_1_5_validation import walk_forward_cv
from src.common.config import Config
from src.math.step_1_2_Q_estimation import estimate_Q_global
from src.math.step_1_4_nll_optimize import optimize_nll, prepare_match_data

log = structlog.get_logger(__name__)

# Paths
_COMMENTARIES_DIR = Path("data/commentaries")
_ODDS_DIR = Path("data/odds_historical")

# sigma_a grid search values
_SIGMA_A_GRID = [0.3, 1.0]

# NLL optimization epochs
_NLL_EPOCHS = 200
_CV_EPOCHS = 100


def _result_code(match: dict) -> str:
    """Get FTR result code from a parsed match."""
    hg = match.get("home_goals", 0)
    ag = match.get("away_goals", 0)
    if hg > ag:
        return "H"
    elif hg < ag:
        return "A"
    return "D"


@dataclass
class _Phase1Result:
    """Internal result container for the CPU-bound computation."""
    go: bool
    Q: np.ndarray | None = None
    b: np.ndarray | None = None
    gamma_H: float = 0.0
    gamma_A: float = 0.0
    delta_H: float = 0.0
    delta_A: float = 0.0
    sigma_a: float = 0.5
    xgb_blob: bytes | None = None
    feature_names: list[str] | None = None
    match_count: int = 0
    brier_score: float = 1.0
    xgb_used: bool = False
    matched_odds: int = 0


def _run_calibration(league_id: str) -> _Phase1Result:
    """CPU-bound calibration pipeline. Runs in a thread to avoid asyncio conflicts."""
    log.info("phase1_start", league_id=league_id)

    # 1. Parse commentaries
    all_matches = parse_commentaries_dir(_COMMENTARIES_DIR)
    league_matches = [m for m in all_matches if m["league_id"] == league_id]
    log.info("matches_filtered", league_id=league_id, count=len(league_matches))

    if len(league_matches) < 50:
        log.warning("insufficient_matches", league_id=league_id, count=len(league_matches))
        return _Phase1Result(go=False)

    # 2. Load odds
    odds_data = load_odds_csv(_ODDS_DIR)
    log.info("odds_loaded", total=len(odds_data))

    # 3. Segment intervals
    intervals_by_match = segment_all_matches(league_matches)
    match_ids = [m["match_id"] for m in league_matches if m["match_id"] in intervals_by_match]
    log.info("intervals_segmented", matches=len(match_ids))

    if len(match_ids) < 50:
        log.warning("insufficient_segmented_matches", count=len(match_ids))
        return _Phase1Result(go=False)

    # Flatten all intervals for Q estimation
    all_intervals = []
    for mid in match_ids:
        all_intervals.extend(intervals_by_match[mid])

    # 4. Estimate Q matrix (step 1.2)
    Q = estimate_Q_global(all_intervals)
    log.info("Q_estimated", Q_shape=Q.shape)

    # 5. Train XGBoost prior (step 1.3)
    matches_ordered = [m for m in league_matches if m["match_id"] in intervals_by_match]
    xgb_result, feature_names, a_H_init, a_A_init = train_xgboost_prior(
        matches_ordered, odds_data, league_id,
    )
    xgb_used = xgb_result is not None
    # Count matched odds
    from src.calibration.team_aliases import normalize_team_name
    matched_odds = 0
    for m in matches_ordered:
        date = m.get("date", "").replace(".", "/")
        h = normalize_team_name(m.get("home_team", ""))
        a = normalize_team_name(m.get("away_team", ""))
        key = f"{date}_{h}_{a}"
        key_swap = f"{date}_{a}_{h}"
        if key in odds_data or key_swap in odds_data:
            matched_odds += 1
    log.info("odds_matching", matched=matched_odds, total=len(matches_ordered), xgb_used=xgb_used)

    # Get actual results for validation
    actual_results = [_result_code(m) for m in matches_ordered]

    # 6 + 7. NLL optimization + validation with sigma_a grid search
    best_sigma_a = _SIGMA_A_GRID[0]
    best_brier = float("inf")
    best_opt_result = None
    best_cv: dict = {}

    for sigma_a in _SIGMA_A_GRID:
        log.info("trying_sigma_a", sigma_a=sigma_a)

        match_data = prepare_match_data(intervals_by_match, match_ids)
        opt_result = optimize_nll(
            match_data, a_H_init, a_A_init,
            sigma_a=sigma_a, num_epochs=_NLL_EPOCHS,
        )

        cv_result = walk_forward_cv(
            intervals_by_match, match_ids,
            a_H_init, a_A_init,
            actual_results, Q,
            sigma_a=sigma_a, num_epochs=_CV_EPOCHS,
            matches_ordered=matches_ordered,
            odds_data=odds_data,
            league_id=league_id,
        )

        overall_brier = cv_result["overall_brier"]
        log.info(
            "sigma_a_result",
            sigma_a=sigma_a,
            brier=round(overall_brier, 4),
            go_nogo=cv_result["go_nogo"],
        )

        if overall_brier < best_brier:
            best_brier = overall_brier
            best_sigma_a = sigma_a
            best_opt_result = opt_result
            best_cv = cv_result

    log.info("grid_search_complete", best_sigma_a=best_sigma_a, best_brier=round(best_brier, 4))

    go_nogo = best_cv.get("go_nogo", "NO-GO")
    log.info("validation_verdict", go_nogo=go_nogo)

    if go_nogo != "GO" or best_opt_result is None:
        log.warning("phase1_no_go", league_id=league_id, brier=round(best_brier, 4))
        return _Phase1Result(go=False, brier_score=best_brier, match_count=len(match_ids),
                             matched_odds=matched_odds, xgb_used=xgb_used)

    # Serialize XGBoost model
    xgb_blob: bytes | None = None
    if xgb_result is not None:
        try:
            model_H, model_A = xgb_result
            buf = io.BytesIO()
            pickle.dump({"home": model_H, "away": model_A}, buf)
            xgb_blob = buf.getvalue()
        except Exception as e:
            log.warning("xgb_serialize_failed", error=str(e))

    gamma_H_val = float(best_opt_result.gamma_H[1]) if len(best_opt_result.gamma_H) > 1 else 0.0
    gamma_A_val = float(best_opt_result.gamma_A[2]) if len(best_opt_result.gamma_A) > 2 else 0.0
    delta_H_val = float(best_opt_result.delta_H[2]) if len(best_opt_result.delta_H) > 2 else 0.0
    delta_A_val = float(best_opt_result.delta_A[2]) if len(best_opt_result.delta_A) > 2 else 0.0

    return _Phase1Result(
        go=True, Q=Q, b=best_opt_result.b,
        gamma_H=gamma_H_val, gamma_A=gamma_A_val,
        delta_H=delta_H_val, delta_A=delta_A_val,
        sigma_a=best_sigma_a, xgb_blob=xgb_blob,
        feature_names=feature_names if xgb_result else None,
        match_count=len(match_ids), brier_score=best_brier,
        xgb_used=xgb_used, matched_odds=matched_odds,
    )


async def save_production_params(
    config: Config,
    league_id: int,
    Q: np.ndarray,
    b: np.ndarray,
    gamma_H: float,
    gamma_A: float,
    delta_H: float,
    delta_A: float,
    sigma_a: float,
    xgb_model_blob: bytes | None,
    feature_mask: list[str] | None,
    match_count: int,
    brier_score: float,
) -> int:
    """INSERT into production_params table. Returns version number."""
    dsn = f"postgresql://{config.db_user}:{config.db_password}@{config.db_host}:{config.db_port}/{config.db_name}"
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute(
                "UPDATE production_params SET is_active = FALSE WHERE league_id = $1",
                league_id,
            )
            version = await conn.fetchval(
                """
                INSERT INTO production_params
                    (league_id, Q, b, gamma_H, gamma_A, delta_H, delta_A,
                     sigma_a, xgb_model_blob, feature_mask, trained_at,
                     match_count, brier_score, is_active)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, TRUE)
                RETURNING version
                """,
                league_id,
                json.dumps(Q.tolist()),
                json.dumps(b.tolist()),
                gamma_H, gamma_A, delta_H, delta_A, sigma_a,
                xgb_model_blob,
                json.dumps(feature_mask) if feature_mask else None,
                datetime.now(timezone.utc),
                match_count, brier_score,
            )
        log.info("params_saved", version=version, league_id=league_id)
        return version
    finally:
        await conn.close()


async def run_phase1(league_id: str, config: Config) -> bool:
    """Full Phase 1 pipeline for a single league.

    CPU-bound work runs in a thread to avoid asyncio/XGBoost conflicts.
    Only DB save uses async.
    """
    # Run CPU-bound calibration in a thread
    result = await asyncio.to_thread(_run_calibration, league_id)

    if not result.go:
        return False

    # Save to DB (async)
    version = await save_production_params(
        config=config,
        league_id=int(league_id),
        Q=result.Q,
        b=result.b,
        gamma_H=result.gamma_H,
        gamma_A=result.gamma_A,
        delta_H=result.delta_H,
        delta_A=result.delta_A,
        sigma_a=result.sigma_a,
        xgb_model_blob=result.xgb_blob,
        feature_mask=result.feature_names,
        match_count=result.match_count,
        brier_score=result.brier_score,
    )

    log.info(
        "phase1_complete",
        league_id=league_id,
        version=version,
        brier=round(result.brier_score, 4),
        match_count=result.match_count,
        matched_odds=result.matched_odds,
        xgb_used=result.xgb_used,
        go_nogo="GO",
    )
    return True
