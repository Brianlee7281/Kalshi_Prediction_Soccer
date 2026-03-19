"""Monte Carlo simulation core — CUDA GPU-accelerated version.

Each CUDA thread simulates one independent match path via thinning
on the MMPP. Uses xoroshiro128p RNG for per-thread random streams.

Falls back gracefully: if no CUDA GPU is available, _HAS_CUDA is False
and callers should use the CPU @njit path in mc_core.py instead.
"""

from __future__ import annotations

import math

import numpy as np

_HAS_CUDA = False
try:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float64,
    )

    if cuda.is_available():
        _HAS_CUDA = True
except Exception:
    pass


if _HAS_CUDA:

    @cuda.jit
    def _mc_simulate_core_cuda(
        t_now,
        T_end,
        S_H,
        S_A,
        state,
        score_diff,
        a_H,
        a_A,
        b,
        gamma_H,
        gamma_A,
        delta_H_pos,
        delta_H_neg,
        delta_A_pos,
        delta_A_neg,
        Q_diag,
        Q_off,
        basis_bounds,
        rng_states,
        results,
        eta_H,
        eta_A,
        eta_H2,
        eta_A2,
        stoppage_1_start,
        stoppage_2_start,
        n_periods,
        N,
    ):
        tid = cuda.grid(1)
        if tid >= N:
            return

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

            # Delta index: each team uses its own perspective
            di_H = sd + 2
            if di_H < 0:
                di_H = 0
            elif di_H > 4:
                di_H = 4
            di_A = -sd + 2
            if di_A < 0:
                di_A = 0
            elif di_A > 4:
                di_A = 4

            # Asymmetric delta lookup
            if sd > 0:  # home leading
                dH = delta_H_pos[di_H]
                dA = delta_A_pos[di_A]
            else:  # trailing or tied
                dH = delta_H_neg[di_H]
                dA = delta_A_neg[di_A]

            # Stoppage time eta multiplier
            eta_h = 0.0
            eta_a = 0.0
            if stoppage_1_start < s < stoppage_1_start + 10.0:
                eta_h = eta_H
                eta_a = eta_A
            elif stoppage_2_start < s < stoppage_2_start + 10.0:
                eta_h = eta_H2
                eta_a = eta_A2

            # Intensities (use math.exp/log for CUDA)
            lam_H = math.exp(a_H + b[bi] + gamma_H[st] + dH + eta_h)
            lam_A = math.exp(a_A + b[bi] + gamma_A[st] + dA + eta_a)
            lam_red = -Q_diag[st]
            lam_total = lam_H + lam_A + lam_red

            if lam_total <= 0.0:
                break

            # Waiting time to next event (exponential)
            u_exp = xoroshiro128p_uniform_float64(rng_states, tid)
            # Clamp to avoid log(0)
            if u_exp <= 0.0:
                u_exp = 1e-300
            dt = -math.log(u_exp) / lam_total
            s_next = s + dt

            # Find next basis boundary or match end
            next_bound = T_end
            n_bounds = n_periods + 1
            for k in range(n_bounds):
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
            u_evt = xoroshiro128p_uniform_float64(rng_states, tid) * lam_total
            if u_evt < lam_H:
                sh += 1
                sd += 1
            elif u_evt < lam_H + lam_A:
                sa += 1
                sd -= 1
            else:
                # Red card transition using normalized Q_off
                cum = 0.0
                r = xoroshiro128p_uniform_float64(rng_states, tid)
                for j in range(4):
                    if j == st:
                        continue
                    cum += Q_off[st, j]
                    if r < cum:
                        st = j
                        break

        results[tid, 0] = sh
        results[tid, 1] = sa

    def mc_simulate_remaining_cuda(
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
        """Host-side launcher for CUDA MC simulation.

        Allocates device memory, launches kernel, copies results back.
        Returns (N, 2) int32 array of [home_goals, away_goals].
        """
        n_periods = len(basis_bounds) - 1

        # Transfer input arrays to device
        d_b = cuda.to_device(np.ascontiguousarray(b, dtype=np.float64))
        d_gamma_H = cuda.to_device(np.ascontiguousarray(gamma_H, dtype=np.float64))
        d_gamma_A = cuda.to_device(np.ascontiguousarray(gamma_A, dtype=np.float64))
        d_delta_H_pos = cuda.to_device(np.ascontiguousarray(delta_H_pos, dtype=np.float64))
        d_delta_H_neg = cuda.to_device(np.ascontiguousarray(delta_H_neg, dtype=np.float64))
        d_delta_A_pos = cuda.to_device(np.ascontiguousarray(delta_A_pos, dtype=np.float64))
        d_delta_A_neg = cuda.to_device(np.ascontiguousarray(delta_A_neg, dtype=np.float64))
        d_Q_diag = cuda.to_device(np.ascontiguousarray(Q_diag, dtype=np.float64))
        d_Q_off = cuda.to_device(np.ascontiguousarray(Q_off, dtype=np.float64))
        d_basis_bounds = cuda.to_device(np.ascontiguousarray(basis_bounds, dtype=np.float64))

        # Allocate output and RNG states on device
        d_results = cuda.device_array((N, 2), dtype=np.int32)
        rng_states = create_xoroshiro128p_states(N, seed=seed)

        # Launch kernel
        threads_per_block = 256
        blocks = math.ceil(N / threads_per_block)

        _mc_simulate_core_cuda[blocks, threads_per_block](
            t_now, T_end, S_H, S_A, state, score_diff,
            a_H, a_A,
            d_b, d_gamma_H, d_gamma_A,
            d_delta_H_pos, d_delta_H_neg, d_delta_A_pos, d_delta_A_neg,
            d_Q_diag, d_Q_off, d_basis_bounds,
            rng_states, d_results,
            eta_H, eta_A, eta_H2, eta_A2,
            stoppage_1_start, stoppage_2_start,
            n_periods, N,
        )

        return d_results.copy_to_host()
