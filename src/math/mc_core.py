"""Monte Carlo simulation core — Numba JIT-compiled.

Pure @njit function for simulating remaining match outcomes using the
MMPP (Markov-Modulated Poisson Process) intensity model.

All inputs/outputs are numpy arrays and scalars — no Python objects,
dicts, lists, or strings allowed inside the JIT-compiled function.

Reference: docs/phase3.md §Logic B: Monte Carlo Pricing
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)  # type: ignore[misc]
def mc_simulate_remaining(
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
    delta_H: np.ndarray,      # shape (5,)
    delta_A: np.ndarray,      # shape (5,)
    Q_diag: np.ndarray,       # shape (4,)
    Q_off: np.ndarray,        # shape (4, 4) normalized transition probs
    basis_bounds: np.ndarray,  # shape (n_basis + 1,)
    N: int,
    seed: int,
) -> np.ndarray:
    """Simulate N remaining-match paths via thinning on the MMPP.

    Returns final_scores array of shape (N, 2) with columns [home, away].

    Uses team-specific gamma + normalized Q_off for red card transitions.
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

            # Intensities
            lam_H = np.exp(a_H + b[bi] + gamma_H[st] + delta_H[di])
            lam_A = np.exp(a_A + b[bi] + gamma_A[st] + delta_A[di])
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
