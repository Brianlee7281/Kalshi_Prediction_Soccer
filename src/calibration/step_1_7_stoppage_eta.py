"""Step 1.7: Estimate stoppage time intensity multipliers (v5).

Compares goal rates in stoppage periods vs normal play to estimate
eta multipliers for first and second half stoppage.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.common.types import IntervalRecord

logger = structlog.get_logger()

_MIN_STOPPAGE_INTERVALS = 50


def estimate_stoppage_eta(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: object,  # OptimizationResult from step_1_4
) -> tuple[float, float, float, float]:
    """Estimate stoppage time intensity multipliers.

    Compares goal rates in first/second half stoppage periods against
    normal play to derive log-ratio eta multipliers.

    Args:
        intervals_by_match: Dict mapping match_id to its intervals.
        opt_result: OptimizationResult from step_1_4 (unused currently,
            reserved for future weighted estimation).

    Returns:
        (eta_H, eta_A, eta_H2, eta_A2) where:
        - eta_H:  1st half stoppage, home intensity multiplier
        - eta_A:  1st half stoppage, away intensity multiplier
        - eta_H2: 2nd half stoppage, home intensity multiplier
        - eta_A2: 2nd half stoppage, away intensity multiplier
    """
    # Accumulators: [exposure_time, home_goals, away_goals]
    stop1_exposure = 0.0
    stop1_home_goals = 0
    stop1_away_goals = 0
    stop1_count = 0

    stop2_exposure = 0.0
    stop2_home_goals = 0
    stop2_away_goals = 0
    stop2_count = 0

    normal_exposure = 0.0
    normal_home_goals = 0
    normal_away_goals = 0

    for ivs in intervals_by_match.values():
        for iv in ivs:
            if iv.is_halftime:
                continue

            duration = iv.t_end - iv.t_start
            if duration <= 0:
                continue

            h_goals = len(iv.home_goal_times)
            a_goals = len(iv.away_goal_times)

            if iv.t_start >= 45.0 and iv.t_end <= 55.0:
                # First-half stoppage
                stop1_exposure += duration
                stop1_home_goals += h_goals
                stop1_away_goals += a_goals
                stop1_count += 1
            elif iv.t_start >= 90.0:
                # Second-half stoppage
                stop2_exposure += duration
                stop2_home_goals += h_goals
                stop2_away_goals += a_goals
                stop2_count += 1
            else:
                # Normal play
                normal_exposure += duration
                normal_home_goals += h_goals
                normal_away_goals += a_goals

    total_stoppage = stop1_count + stop2_count

    logger.info(
        "stoppage_eta_stats",
        stop1_count=stop1_count,
        stop1_exposure=round(stop1_exposure, 1),
        stop2_count=stop2_count,
        stop2_exposure=round(stop2_exposure, 1),
        normal_exposure=round(normal_exposure, 1),
        total_stoppage_intervals=total_stoppage,
    )

    if total_stoppage < _MIN_STOPPAGE_INTERVALS:
        logger.warning(
            "stoppage_eta_insufficient_data",
            total_stoppage_intervals=total_stoppage,
            min_required=_MIN_STOPPAGE_INTERVALS,
        )
        return (0.0, 0.0, 0.0, 0.0)

    # Normal play rates (goals per minute)
    normal_home_rate = normal_home_goals / normal_exposure if normal_exposure > 0 else 0.0
    normal_away_rate = normal_away_goals / normal_exposure if normal_exposure > 0 else 0.0

    # First-half stoppage rates
    stop1_home_rate = stop1_home_goals / stop1_exposure if stop1_exposure > 0 else 0.0
    stop1_away_rate = stop1_away_goals / stop1_exposure if stop1_exposure > 0 else 0.0

    # Second-half stoppage rates
    stop2_home_rate = stop2_home_goals / stop2_exposure if stop2_exposure > 0 else 0.0
    stop2_away_rate = stop2_away_goals / stop2_exposure if stop2_exposure > 0 else 0.0

    # Compute eta as log-ratio vs normal play
    eta_H = _log_ratio(stop1_home_rate, normal_home_rate)
    eta_A = _log_ratio(stop1_away_rate, normal_away_rate)
    eta_H2 = _log_ratio(stop2_home_rate, normal_home_rate)
    eta_A2 = _log_ratio(stop2_away_rate, normal_away_rate)

    # Clamp to [-1.0, 1.0]
    eta_H = float(np.clip(eta_H, -1.0, 1.0))
    eta_A = float(np.clip(eta_A, -1.0, 1.0))
    eta_H2 = float(np.clip(eta_H2, -1.0, 1.0))
    eta_A2 = float(np.clip(eta_A2, -1.0, 1.0))

    logger.info(
        "stoppage_eta_result",
        eta_H=round(eta_H, 4),
        eta_A=round(eta_A, 4),
        eta_H2=round(eta_H2, 4),
        eta_A2=round(eta_A2, 4),
    )

    return (eta_H, eta_A, eta_H2, eta_A2)


def _log_ratio(stoppage_rate: float, normal_rate: float) -> float:
    """Compute log(stoppage_rate / normal_rate), returning 0.0 if either is zero."""
    if stoppage_rate > 0 and normal_rate > 0:
        return float(np.log(stoppage_rate / normal_rate))
    return 0.0
