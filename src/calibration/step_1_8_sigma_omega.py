"""Step 1.8: Estimate EKF process noise σ²_ω (v5).

Measures within-match team strength drift by comparing first-half
vs second-half implied intensities.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.common.types import IntervalRecord

logger = structlog.get_logger()

_MIN_MATCHES = 30
_EPSILON = 0.01
_DEFAULT_SIGMA_OMEGA_SQ = 0.01


def estimate_sigma_omega_sq(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: object,  # OptimizationResult from step_1_4
) -> float:
    """Estimate EKF process noise σ²_ω.

    Method: For each match, estimate implied a_H using first-half goals only
    vs second-half goals only. σ²_ω = Var(Δa) / 45 minutes.

    Args:
        intervals_by_match: Dict mapping match_id to its intervals.
        opt_result: OptimizationResult from step_1_4 (reserved for future
            weighted estimation).

    Returns:
        Estimated σ²_ω, clamped to [0.001, 0.1].
    """
    delta_a_list: list[float] = []

    for match_id, ivs in intervals_by_match.items():
        first_half_exposure = 0.0
        first_half_goals = 0
        second_half_exposure = 0.0
        second_half_goals = 0

        for iv in ivs:
            if iv.is_halftime:
                continue

            duration = iv.t_end - iv.t_start
            if duration <= 0:
                continue

            h_goals = len(iv.home_goal_times)

            if iv.t_end <= 45.0:
                first_half_exposure += duration
                first_half_goals += h_goals
            elif iv.t_start >= 45.0:
                second_half_exposure += duration
                second_half_goals += h_goals

        if first_half_exposure <= 0 or second_half_exposure <= 0:
            continue

        first_rate = first_half_goals / first_half_exposure
        second_rate = second_half_goals / second_half_exposure

        # Use epsilon floor to avoid log(0)
        if first_rate <= 0:
            first_rate = _EPSILON
        if second_rate <= 0:
            second_rate = _EPSILON

        implied_a_first = np.log(first_rate)
        implied_a_second = np.log(second_rate)
        delta_a_list.append(implied_a_second - implied_a_first)

    logger.info(
        "sigma_omega_estimation",
        matches_with_valid_delta=len(delta_a_list),
    )

    if len(delta_a_list) < _MIN_MATCHES:
        logger.warning(
            "sigma_omega_insufficient_data",
            valid_matches=len(delta_a_list),
            min_required=_MIN_MATCHES,
        )
        return _DEFAULT_SIGMA_OMEGA_SQ

    delta_a = np.array(delta_a_list)
    sigma_omega_sq = float(np.var(delta_a)) / 45.0

    sigma_omega_sq = float(np.clip(sigma_omega_sq, 0.001, 0.1))

    logger.info(
        "sigma_omega_result",
        sigma_omega_sq=round(sigma_omega_sq, 6),
        delta_a_mean=round(float(np.mean(delta_a)), 4),
        delta_a_std=round(float(np.std(delta_a)), 4),
    )

    return sigma_omega_sq
