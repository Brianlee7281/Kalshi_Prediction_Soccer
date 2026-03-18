"""Step 1.6: Estimate asymmetric score-state effects (v5).

Splits historical intervals by sign of delta_S to estimate separate
delta arrays for when a team is leading vs trailing.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.common.types import IntervalRecord

logger = structlog.get_logger()

_NUM_DS_BINS = 5
_REF_BIN = 2  # ΔS = 0 is the reference bin
_MIN_INTERVALS = 20  # minimum intervals per bin before falling back to symmetric


def _ds_to_bin(delta_S: int) -> int:
    """Map score difference to bin index [0, 4]."""
    if delta_S <= -2:
        return 0
    if delta_S >= 2:
        return 4
    return delta_S + 2


def estimate_asymmetric_delta(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: object,  # OptimizationResult from step_1_4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate asymmetric score-state effects from historical intervals.

    Splits intervals by sign of delta_S, fits MLE for each subset.

    Args:
        intervals_by_match: Dict mapping match_id to its intervals.
        opt_result: OptimizationResult from step_1_4. Must have
            delta_H and delta_A attributes (numpy arrays of shape (5,)).

    Returns:
        (delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg), each shape (5,).
        - delta_H_pos: home intensity effect when home is leading (sd > 0)
        - delta_H_neg: home intensity effect when trailing or tied (sd <= 0)
        - delta_A_pos: away intensity effect when home is leading (sd > 0)
        - delta_A_neg: away intensity effect when trailing or tied (sd <= 0)
    """
    sym_delta_H: np.ndarray = opt_result.delta_H  # type: ignore[attr-defined]
    sym_delta_A: np.ndarray = opt_result.delta_A  # type: ignore[attr-defined]

    # Flatten all intervals, skip halftime
    all_intervals: list[IntervalRecord] = []
    for ivs in intervals_by_match.values():
        for iv in ivs:
            if not iv.is_halftime:
                all_intervals.append(iv)

    # Partition into leading (delta_S > 0) and trailing_or_tied (delta_S <= 0)
    leading: list[IntervalRecord] = []
    trailing_or_tied: list[IntervalRecord] = []
    for iv in all_intervals:
        if iv.delta_S > 0:
            leading.append(iv)
        else:
            trailing_or_tied.append(iv)

    logger.info(
        "asymmetric_delta_partition",
        total_intervals=len(all_intervals),
        leading=len(leading),
        trailing_or_tied=len(trailing_or_tied),
    )

    delta_H_pos = _estimate_delta_for_partition(leading, "home", sym_delta_H)
    delta_A_pos = _estimate_delta_for_partition(leading, "away", sym_delta_A)
    delta_H_neg = _estimate_delta_for_partition(trailing_or_tied, "home", sym_delta_H)
    delta_A_neg = _estimate_delta_for_partition(trailing_or_tied, "away", sym_delta_A)

    logger.info(
        "asymmetric_delta_result",
        delta_H_pos=delta_H_pos.tolist(),
        delta_H_neg=delta_H_neg.tolist(),
        delta_A_pos=delta_A_pos.tolist(),
        delta_A_neg=delta_A_neg.tolist(),
    )

    return delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg


def _estimate_delta_for_partition(
    intervals: list[IntervalRecord],
    team: str,
    sym_fallback: np.ndarray,
) -> np.ndarray:
    """Estimate delta array for one partition (leading or trailing_or_tied).

    For each bin, computes goal rate = goals / exposure_time, then
    delta[bin] = log(rate[bin] / rate[ref_bin]).

    Falls back to symmetric delta if a bin has fewer than _MIN_INTERVALS.

    Args:
        intervals: Subset of intervals for this partition.
        team: "home" or "away".
        sym_fallback: Symmetric delta array shape (5,) for fallback.

    Returns:
        Delta array of shape (5,), clamped to [-0.5, 0.5].
    """
    exposure = np.zeros(_NUM_DS_BINS)
    goals = np.zeros(_NUM_DS_BINS)
    counts = np.zeros(_NUM_DS_BINS, dtype=np.int64)

    for iv in intervals:
        bi = _ds_to_bin(iv.delta_S)
        duration = iv.t_end - iv.t_start
        if duration <= 0:
            continue

        counts[bi] += 1
        exposure[bi] += duration

        if team == "home":
            goals[bi] += len(iv.home_goal_times)
        else:
            goals[bi] += len(iv.away_goal_times)

    # Compute rates (goals per minute)
    rates = np.zeros(_NUM_DS_BINS)
    for i in range(_NUM_DS_BINS):
        if exposure[i] > 0:
            rates[i] = goals[i] / exposure[i]

    # Reference rate (bin 2, ΔS = 0)
    ref_rate = rates[_REF_BIN]

    # Compute log-ratio relative to reference bin
    delta = np.zeros(_NUM_DS_BINS)
    for i in range(_NUM_DS_BINS):
        if counts[i] < _MIN_INTERVALS:
            delta[i] = sym_fallback[i]
            logger.debug(
                "asymmetric_delta_fallback",
                team=team,
                bin=i,
                count=int(counts[i]),
                reason="insufficient_intervals",
            )
        elif rates[i] > 0 and ref_rate > 0:
            delta[i] = np.log(rates[i] / ref_rate)
        else:
            delta[i] = 0.0

    # Clamp to [-0.5, 0.5]
    delta = np.clip(delta, -0.5, 0.5)

    return delta
