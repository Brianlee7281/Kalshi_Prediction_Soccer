"""Step 1.2 — Q Matrix Estimation.

Estimates the 4×4 Markov chain generator matrix Q for red-card
state transitions from historical interval data.

State space:
    0: 11v11 (normal play)
    1: 10v11 (home sent off)
    2: 11v10 (away sent off)
    3: 10v10 (both sent off)

Reference: docs/phase1.md Step 1.2
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.common.types import IntervalRecord

_NUM_STATES = 4

# Default shrinkage threshold in match-minutes per ΔS bin.
_DEFAULT_T_THRESHOLD = 5000.0


# ---------------------------------------------------------------------------
# Global Q estimation
# ---------------------------------------------------------------------------


def estimate_Q_global(
    intervals: list[IntervalRecord],
) -> npt.NDArray[np.float64]:
    """Estimate the 4×4 generator matrix Q from interval data.

    Counts red-card transitions and total dwell time per state across
    all matches, then computes off-diagonal rates and sets the diagonal
    so each row sums to zero.

    Args:
        intervals: IntervalRecord list from Step 1.1.

    Returns:
        4×4 numpy array satisfying the generator property (rows sum to 0).
    """
    N = np.zeros((_NUM_STATES, _NUM_STATES), dtype=np.float64)
    T = np.zeros(_NUM_STATES, dtype=np.float64)

    for iv in intervals:
        if iv.is_halftime:
            continue

        duration = iv.t_end - iv.t_start
        T[iv.state_X] += duration

        for rc in iv.red_card_transitions:
            N[rc.from_state, rc.to_state] += 1

    return _build_Q_from_counts(N, T)


# ---------------------------------------------------------------------------
# Score-stratified Q with shrinkage
# ---------------------------------------------------------------------------

# ΔS bins: ≤-2 → 0, -1 → 1, 0 → 2, +1 → 3, ≥+2 → 4
_NUM_DS_BINS = 5


def _ds_to_bin(delta_S: int) -> int:
    """Map a score difference to a ΔS bin index in [0, 4]."""
    if delta_S <= -2:
        return 0
    if delta_S >= 2:
        return 4
    return delta_S + 2  # -1→1, 0→2, 1→3


def estimate_Q_by_delta_S(
    intervals: list[IntervalRecord],
    *,
    T_threshold: float = _DEFAULT_T_THRESHOLD,
) -> dict[int, npt.NDArray[np.float64]]:
    """Estimate per-ΔS-bin Q matrices with hierarchical shrinkage.

    Each bin's empirical Q is shrunk toward the global Q based on
    total dwell time in that bin relative to *T_threshold*.

    Args:
        intervals: IntervalRecord list from Step 1.1.
        T_threshold: Minutes of dwell time at which the bin-level
            estimate receives full weight (no shrinkage).

    Returns:
        Dict mapping bin index (0–4) to a 4×4 shrunk Q matrix.
    """
    N_bin: dict[int, npt.NDArray[np.float64]] = {
        b: np.zeros((_NUM_STATES, _NUM_STATES), dtype=np.float64)
        for b in range(_NUM_DS_BINS)
    }
    T_bin: dict[int, npt.NDArray[np.float64]] = {
        b: np.zeros(_NUM_STATES, dtype=np.float64)
        for b in range(_NUM_DS_BINS)
    }

    for iv in intervals:
        if iv.is_halftime:
            continue

        b = _ds_to_bin(iv.delta_S)
        duration = iv.t_end - iv.t_start
        T_bin[b][iv.state_X] += duration

        for rc in iv.red_card_transitions:
            N_bin[b][rc.from_state, rc.to_state] += 1

    Q_global = estimate_Q_global(intervals)

    result: dict[int, npt.NDArray[np.float64]] = {}
    for b in range(_NUM_DS_BINS):
        Q_emp = _build_Q_from_counts(N_bin[b], T_bin[b])
        total_dwell = float(np.sum(T_bin[b]))
        w = min(1.0, total_dwell / T_threshold)
        result[b] = w * Q_emp + (1.0 - w) * Q_global

    return result


# ---------------------------------------------------------------------------
# State-3 sparsity: additivity fill
# ---------------------------------------------------------------------------


def apply_state3_additivity(
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fill sparse state-3 transitions using the additivity assumption.

    When 10v10 data is too sparse, approximate:
        q(1→3) ≈ q(0→2)   (away dismissal rate unchanged)
        q(2→3) ≈ q(0→1)   (home dismissal rate unchanged)

    The diagonal is recomputed after patching.

    Args:
        Q: 4×4 generator matrix (modified in-place and returned).

    Returns:
        The same array with state-3 transitions patched.
    """
    Q_out = Q.copy()
    Q_out[1, 3] = Q_out[0, 2]  # away red while home already down
    Q_out[2, 3] = Q_out[0, 1]  # home red while away already down

    # Recompute diagonals for rows 1 and 2
    for i in (1, 2):
        Q_out[i, i] = -np.sum(Q_out[i, :]) + Q_out[i, i]  # subtract new off-diag
        Q_out[i, i] = 0.0
        Q_out[i, i] = -np.sum(Q_out[i, :])

    return Q_out


# ---------------------------------------------------------------------------
# Normalized off-diagonal (Phase 3 MC sampling)
# ---------------------------------------------------------------------------


def normalize_Q_off_diagonal(
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize off-diagonal entries to transition probabilities.

    Each row's off-diagonal entries are divided by their sum so that
    they form a valid probability distribution for sampling which
    state transition occurs on a dismissal event.

    Args:
        Q: 4×4 generator matrix.

    Returns:
        4×4 array where off-diagonal entries per row sum to 1.0
        (diagonal is 0).
    """
    Q_off = np.zeros_like(Q)
    for i in range(_NUM_STATES):
        row_sum = -Q[i, i]
        if row_sum > 0:
            for j in range(_NUM_STATES):
                if i != j:
                    Q_off[i, j] = Q[i, j] / row_sum
    return Q_off


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_Q_from_counts(
    N: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build a generator matrix from transition counts and dwell times."""
    Q = np.zeros((_NUM_STATES, _NUM_STATES), dtype=np.float64)
    for i in range(_NUM_STATES):
        if T[i] > 0:
            for j in range(_NUM_STATES):
                if i != j:
                    Q[i, j] = N[i, j] / T[i]
        Q[i, i] = -np.sum(Q[i, :])
    return Q
