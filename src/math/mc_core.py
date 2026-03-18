"""Monte Carlo simulation core — Numba JIT-compiled with optional GPU.

CPU path: Pure @njit functions for simulating remaining match outcomes
using the MMPP (Markov-Modulated Poisson Process) intensity model.

GPU path: When a CUDA GPU is available, simulations are dispatched to
mc_core_cuda.py where each thread handles one independent path.

All inputs/outputs are numpy arrays and scalars — no Python objects,
dicts, lists, or strings allowed inside the JIT-compiled function.

Reference: docs/phase3.md §Logic B: Monte Carlo Pricing
"""

from __future__ import annotations

import numpy as np
from numba import njit

# GPU acceleration (optional — falls back to CPU @njit if unavailable)
_USE_GPU = False
try:
    from src.math.mc_core_cuda import _HAS_CUDA, mc_simulate_remaining_cuda

    _USE_GPU = _HAS_CUDA
except ImportError:
    pass


@njit(cache=True)  # type: ignore[misc]
def _mc_simulate_core(
    t_now: float,
    T_end: float,
    S_H: int,
    S_A: int,
    state: int,
    score_diff: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,            # shape (n_basis,)
    gamma_H: np.ndarray,      # shape (4,)
    gamma_A: np.ndarray,      # shape (4,)
    delta_H_pos: np.ndarray,  # shape (5,), home when leading
    delta_H_neg: np.ndarray,  # shape (5,), home when trailing/tied
    delta_A_pos: np.ndarray,  # shape (5,), away when trailing/tied
    delta_A_neg: np.ndarray,  # shape (5,), away when leading
    Q_diag: np.ndarray,       # shape (4,)
    Q_off: np.ndarray,        # shape (4, 4) normalized transition probs
    basis_bounds: np.ndarray,  # shape (n_basis + 1,)
    N: int,
    seed: int,
    eta_H: float,
    eta_A: float,
    eta_H2: float,
    eta_A2: float,
    stoppage_1_start: float,
    stoppage_2_start: float,
) -> np.ndarray:
    """Simulate N remaining-match paths via thinning on the MMPP (CPU).

    Returns final_scores array of shape (N, 2) with columns [home, away].

    Uses team-specific gamma + normalized Q_off for red card transitions.
    Asymmetric delta arrays allow different score-state effects depending
    on whether a team is leading or trailing.
    Eta stoppage multipliers boost intensity during stoppage time windows.
    Deterministic given the same seed for reproducibility.
    """
    np.random.seed(seed)
    results = np.empty((N, 2), dtype=np.int32)
    n_periods = len(basis_bounds) - 1

    for sim in range(N):
        s = t_now
        sh = S_H
        sa = S_A
        st = state
        sd = score_diff

        while s < T_end:
            # Current basis index
            bi = 0
            for k in range(n_periods):
                if s >= basis_bounds[k] and s < basis_bounds[k + 1]:
                    bi = k
                    break

            # Delta index: ΔS -> {0: ≤-2, 1: -1, 2: 0, 3: +1, 4: ≥+2}
            di = sd + 2
            if di < 0:
                di = 0
            elif di > 4:
                di = 4

            # Asymmetric delta lookup
            if sd > 0:  # home leading
                dH = delta_H_pos[di]
                dA = delta_A_pos[di]
            else:  # trailing or tied
                dH = delta_H_neg[di]
                dA = delta_A_neg[di]

            # Stoppage time eta multiplier
            eta_h = 0.0
            eta_a = 0.0
            if stoppage_1_start < s < stoppage_1_start + 10.0:
                eta_h = eta_H
                eta_a = eta_A
            elif stoppage_2_start < s < stoppage_2_start + 10.0:
                eta_h = eta_H2
                eta_a = eta_A2

            # Intensities
            lam_H = np.exp(a_H + b[bi] + gamma_H[st] + dH + eta_h)
            lam_A = np.exp(a_A + b[bi] + gamma_A[st] + dA + eta_a)
            lam_red = -Q_diag[st]
            lam_total = lam_H + lam_A + lam_red

            if lam_total <= 0.0:
                break

            # Waiting time to next event (exponential)
            dt = -np.log(np.random.random()) / lam_total
            s_next = s + dt

            # Find next basis boundary or match end
            next_bound = T_end
            for k in range(len(basis_bounds)):
                if basis_bounds[k] > s:
                    if basis_bounds[k] < next_bound:
                        next_bound = basis_bounds[k]
                    break

            # If event falls beyond boundary, advance to boundary
            if s_next >= next_bound:
                s = next_bound
                continue

            s = s_next

            # Sample event type
            u = np.random.random() * lam_total
            if u < lam_H:
                sh += 1
                sd += 1
            elif u < lam_H + lam_A:
                sa += 1
                sd -= 1
            else:
                # Red card transition using normalized Q_off
                cum = 0.0
                r = np.random.random()
                for j in range(4):
                    if j == st:
                        continue
                    cum += Q_off[st, j]
                    if r < cum:
                        st = j
                        break

        results[sim, 0] = sh
        results[sim, 1] = sa

    return results


@njit(cache=True)  # type: ignore[misc]
def _mc_simulate_remaining_cpu(
    t_now: float,
    T_end: float,
    S_H: int,
    S_A: int,
    state: int,
    score_diff: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma_H: np.ndarray,
    gamma_A: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int,
    seed: int,
) -> np.ndarray:
    """CPU path — symmetric deltas, no stoppage eta."""
    return _mc_simulate_core(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A,
        delta_H, delta_H, delta_A, delta_A,
        Q_diag, Q_off, basis_bounds, N, seed,
        0.0, 0.0, 0.0, 0.0,
        45.0, 90.0,
    )


@njit(cache=True)  # type: ignore[misc]
def _mc_simulate_remaining_v5_cpu(
    t_now: float,
    T_end: float,
    S_H: int,
    S_A: int,
    state: int,
    score_diff: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma_H: np.ndarray,
    gamma_A: np.ndarray,
    delta_H_pos: np.ndarray,
    delta_H_neg: np.ndarray,
    delta_A_pos: np.ndarray,
    delta_A_neg: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int,
    seed: int,
    eta_H: float,
    eta_A: float,
    eta_H2: float,
    eta_A2: float,
    stoppage_1_start: float,
    stoppage_2_start: float,
) -> np.ndarray:
    """CPU path — asymmetric deltas + stoppage eta."""
    return _mc_simulate_core(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A,
        delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
        Q_diag, Q_off, basis_bounds, N, seed,
        eta_H, eta_A, eta_H2, eta_A2,
        stoppage_1_start, stoppage_2_start,
    )


# ---------------------------------------------------------------------------
# Public API — dispatches to GPU when available, CPU otherwise
# ---------------------------------------------------------------------------

def mc_simulate_remaining(
    t_now: float,
    T_end: float,
    S_H: int,
    S_A: int,
    state: int,
    score_diff: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma_H: np.ndarray,
    gamma_A: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int,
    seed: int,
) -> np.ndarray:
    """Backward-compatible wrapper — symmetric deltas, no stoppage eta.

    Dispatches to GPU when a CUDA device is available, otherwise
    falls back to CPU @njit path. Produces statistically equivalent
    results (not bit-identical due to different RNG streams).
    """
    if _USE_GPU:
        return mc_simulate_remaining_cuda(
            t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
            b, gamma_H, gamma_A,
            delta_H, delta_H, delta_A, delta_A,  # symmetric: pos=neg
            Q_diag, Q_off, basis_bounds, N, seed,
            0.0, 0.0, 0.0, 0.0,  # no eta
            45.0, 90.0,  # default stoppage starts
        )
    return _mc_simulate_remaining_cpu(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A, delta_H, delta_A,
        Q_diag, Q_off, basis_bounds, N, seed,
    )


def mc_simulate_remaining_v5(
    t_now: float,
    T_end: float,
    S_H: int,
    S_A: int,
    state: int,
    score_diff: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma_H: np.ndarray,
    gamma_A: np.ndarray,
    delta_H_pos: np.ndarray,
    delta_H_neg: np.ndarray,
    delta_A_pos: np.ndarray,
    delta_A_neg: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int,
    seed: int,
    eta_H: float,
    eta_A: float,
    eta_H2: float,
    eta_A2: float,
    stoppage_1_start: float,
    stoppage_2_start: float,
) -> np.ndarray:
    """v5 entry point with asymmetric deltas and stoppage eta support.

    Dispatches to GPU when a CUDA device is available.
    """
    if _USE_GPU:
        return mc_simulate_remaining_cuda(
            t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
            b, gamma_H, gamma_A,
            delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
            Q_diag, Q_off, basis_bounds, N, seed,
            eta_H, eta_A, eta_H2, eta_A2,
            stoppage_1_start, stoppage_2_start,
        )
    return _mc_simulate_remaining_v5_cpu(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A,
        delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
        Q_diag, Q_off, basis_bounds, N, seed,
        eta_H, eta_A, eta_H2, eta_A2,
        stoppage_1_start, stoppage_2_start,
    )
