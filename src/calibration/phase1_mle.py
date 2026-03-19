"""Phase 1 MLE: estimate b, delta, gamma from historical goal times.

Sequential estimation via L-BFGS on the Poisson process log-likelihood.
Vectorized: all matches are pre-processed into flat numpy arrays of
(basis_period, delta_S_bin, team, duration) tuples, then LL is computed
in one vectorized pass — ~1000x faster than per-match Python loops.

Identifiability constraints:
  b[0] = 0   (first period is reference)
  delta_H[2] = delta_A[2] = 0  (score-tied is reference)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ── Data types ────────────────────────────────────────────────────

@dataclass
class MatchData:
    """Pre-processed match for MLE."""
    a_H: float
    a_A: float
    goal_times_home: list[float]  # minute + 0.5
    goal_times_away: list[float]
    T: float = 93.0


@dataclass
class Phase1MLEResult:
    b: np.ndarray            # shape (8,), b[0] = 0.0
    delta_H: np.ndarray      # shape (5,), delta_H[2] = 0.0
    delta_A: np.ndarray      # shape (5,), delta_A[2] = 0.0
    gamma_H: float
    gamma_A: float
    train_LL: float
    val_LL: float
    n_matches_train: int
    n_matches_val: int
    rounds_completed: int = 0


# ── Precomputation: flatten all matches into arrays ───────────────

@dataclass
class BatchedData:
    """All match segments flattened into arrays for vectorized LL.

    Interval arrays (length = total segments across all matches):
        iv_a_H, iv_a_A: match-level log-intensities
        iv_bi: basis period index (0-7)
        iv_di_H, iv_di_A: delta_S bin index (0-4) for home/away
        iv_width: segment duration in minutes

    Goal arrays (length = total goals across all matches):
        g_a: match-level log-intensity for scoring team
        g_bi: basis period index at goal time
        g_di: delta_S bin index at goal time
        g_is_home: 1 if home goal, 0 if away
    """
    iv_a_H: np.ndarray
    iv_a_A: np.ndarray
    iv_bi: np.ndarray    # int
    iv_di_H: np.ndarray  # int
    iv_di_A: np.ndarray  # int
    iv_width: np.ndarray

    g_a: np.ndarray      # a_H or a_A depending on team
    g_bi: np.ndarray     # int
    g_di: np.ndarray     # int, delta index for scoring team
    g_is_home: np.ndarray  # bool


def precompute_batch(
    matches: list[MatchData],
    basis_bounds: np.ndarray,
) -> BatchedData:
    """Flatten all matches into vectorized arrays.

    For each match, walk through the timeline splitting at goal events
    and basis_bounds boundaries, recording (bi, di, width) for intervals
    and (bi, di, team) for goals.
    """
    iv_a_H_list: list[float] = []
    iv_a_A_list: list[float] = []
    iv_bi_list: list[int] = []
    iv_di_H_list: list[int] = []
    iv_di_A_list: list[int] = []
    iv_width_list: list[float] = []

    g_a_list: list[float] = []
    g_bi_list: list[int] = []
    g_di_list: list[int] = []
    g_is_home_list: list[int] = []

    n_bounds = len(basis_bounds)

    for m in matches:
        # Sort all goals, clip to [0, T] to handle stoppage-time goals
        T = min(m.T, float(basis_bounds[-1]))
        all_goals = [(min(t, T - 0.01), 1) for t in m.goal_times_home] + \
                    [(min(t, T - 0.01), 0) for t in m.goal_times_away]
        all_goals.sort()

        score_H, score_A = 0, 0
        prev_t = 0.0

        for g_t, g_is_home in all_goals:
            if g_t > prev_t:
                # Add interval segments [prev_t, g_t] split at basis_bounds
                _add_segments(prev_t, g_t, m.a_H, m.a_A,
                              score_H, score_A, basis_bounds, n_bounds,
                              iv_a_H_list, iv_a_A_list, iv_bi_list,
                              iv_di_H_list, iv_di_A_list, iv_width_list)

            # Record goal event
            ds = score_H - score_A
            bi = _basis_idx(g_t, basis_bounds, n_bounds)
            if g_is_home:
                di = min(4, max(0, ds + 2))
                g_a_list.append(m.a_H)
                g_di_list.append(di)
            else:
                di = min(4, max(0, -ds + 2))
                g_a_list.append(m.a_A)
                g_di_list.append(di)
            g_bi_list.append(bi)
            g_is_home_list.append(g_is_home)

            if g_is_home:
                score_H += 1
            else:
                score_A += 1
            prev_t = g_t

        # Final segment [last_goal, T]
        if prev_t < m.T:
            _add_segments(prev_t, m.T, m.a_H, m.a_A,
                          score_H, score_A, basis_bounds, n_bounds,
                          iv_a_H_list, iv_a_A_list, iv_bi_list,
                          iv_di_H_list, iv_di_A_list, iv_width_list)

    return BatchedData(
        iv_a_H=np.array(iv_a_H_list, dtype=np.float64),
        iv_a_A=np.array(iv_a_A_list, dtype=np.float64),
        iv_bi=np.array(iv_bi_list, dtype=np.int32),
        iv_di_H=np.array(iv_di_H_list, dtype=np.int32),
        iv_di_A=np.array(iv_di_A_list, dtype=np.int32),
        iv_width=np.array(iv_width_list, dtype=np.float64),
        g_a=np.array(g_a_list, dtype=np.float64),
        g_bi=np.array(g_bi_list, dtype=np.int32),
        g_di=np.array(g_di_list, dtype=np.int32),
        g_is_home=np.array(g_is_home_list, dtype=np.int32),
    )


def _basis_idx(t: float, bounds: np.ndarray, n: int) -> int:
    for i in range(n - 1):
        if t < bounds[i + 1]:
            return i
    return n - 2


def _add_segments(
    t_lo: float, t_hi: float,
    a_H: float, a_A: float,
    score_H: int, score_A: int,
    bounds: np.ndarray, n_bounds: int,
    iv_a_H: list, iv_a_A: list, iv_bi: list,
    iv_di_H: list, iv_di_A: list, iv_width: list,
) -> None:
    """Add interval segments split at basis_bounds."""
    ds = score_H - score_A
    di_H = min(4, max(0, ds + 2))
    di_A = min(4, max(0, -ds + 2))
    # Clip to [0, T_max] to avoid infinite loops from stoppage-time goals
    t_lo = max(0.0, min(t_lo, bounds[-1]))
    t_hi = max(0.0, min(t_hi, bounds[-1]))
    cur = t_lo
    while cur < t_hi - 1e-9:
        bi = _basis_idx(cur, bounds, n_bounds)
        nxt = min(t_hi, bounds[bi + 1] if bi + 1 < n_bounds else t_hi)
        if nxt <= cur + 1e-9:
            break  # safety: no progress possible
        w = nxt - cur
        if w > 1e-9:
            iv_a_H.append(a_H)
            iv_a_A.append(a_A)
            iv_bi.append(bi)
            iv_di_H.append(di_H)
            iv_di_A.append(di_A)
            iv_width.append(w)
        cur = nxt


# ── Vectorized log-likelihood ─────────────────────────────────────

def total_ll_vec(
    batch: BatchedData,
    b: np.ndarray,
    gamma_H: float,
    gamma_A: float,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
) -> float:
    """Vectorized total LL across all precomputed segments and goals.

    ~1000x faster than per-match Python iteration.
    """
    # ── Integral term (intervals) ─────────────────────────────
    # log_lam_H = a_H + b[bi] + gamma_H + delta_H[di_H]
    log_lam_H = batch.iv_a_H + b[batch.iv_bi] + gamma_H + delta_H[batch.iv_di_H]
    log_lam_A = batch.iv_a_A + b[batch.iv_bi] + gamma_A + delta_A[batch.iv_di_A]
    integral = np.sum((np.exp(log_lam_H) + np.exp(log_lam_A)) * batch.iv_width)

    # ── Point process term (goals) ────────────────────────────
    if len(batch.g_a) == 0:
        return -integral

    # For each goal, log(lambda) = a + b[bi] + gamma + delta[di]
    g_gamma = np.where(batch.g_is_home, gamma_H, gamma_A)
    g_delta = np.where(batch.g_is_home, delta_H[batch.g_di], delta_A[batch.g_di])
    log_lam_goals = batch.g_a + b[batch.g_bi] + g_gamma + g_delta
    goal_ll = np.sum(log_lam_goals)  # sum of log(lambda) at goal times

    return float(goal_ll - integral)


# ── Round 1: Estimate b ──────────────────────────────────────────

def estimate_b(
    batch: BatchedData,
    gamma_H: float,
    gamma_A: float,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    b_init: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate b[1..7] via L-BFGS, with b[0] = 0 fixed."""
    x0 = np.zeros(7) if b_init is None else b_init[1:].copy()

    def neg_ll(x: np.ndarray) -> float:
        b_full = np.zeros(8)
        b_full[1:] = x
        return -total_ll_vec(batch, b_full, gamma_H, gamma_A, delta_H, delta_A)

    result = minimize(neg_ll, x0, method="L-BFGS-B",
                      bounds=[(-1.0, 1.0)] * 7,
                      options={"maxiter": 200, "ftol": 1e-8})
    b_out = np.zeros(8)
    b_out[1:] = result.x
    return b_out


# ── Round 2: Estimate delta ──────────────────────────────────────

def estimate_delta(
    batch: BatchedData,
    b: np.ndarray,
    gamma_H: float,
    gamma_A: float,
    delta_init_H: np.ndarray | None = None,
    delta_init_A: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate delta_H[0,1,3,4] and delta_A[0,1,3,4], index 2 = 0."""
    if delta_init_H is not None and delta_init_A is not None:
        x0 = np.array([
            delta_init_H[0], delta_init_H[1], delta_init_H[3], delta_init_H[4],
            delta_init_A[0], delta_init_A[1], delta_init_A[3], delta_init_A[4],
        ])
    else:
        x0 = np.zeros(8)

    def neg_ll(x: np.ndarray) -> float:
        dH = np.array([x[0], x[1], 0.0, x[2], x[3]])
        dA = np.array([x[4], x[5], 0.0, x[6], x[7]])
        return -total_ll_vec(batch, b, gamma_H, gamma_A, dH, dA)

    result = minimize(neg_ll, x0, method="L-BFGS-B",
                      bounds=[(-0.8, 0.8)] * 8,
                      options={"maxiter": 200, "ftol": 1e-8})

    dH = np.array([result.x[0], result.x[1], 0.0, result.x[2], result.x[3]])
    dA = np.array([result.x[4], result.x[5], 0.0, result.x[6], result.x[7]])
    return dH, dA


# ── Round 3: Estimate gamma ──────────────────────────────────────

def estimate_gamma(
    batch: BatchedData,
    b: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
) -> tuple[float, float]:
    """Estimate gamma_H, gamma_A (2 free params)."""
    x0 = np.array([0.0, 0.0])

    def neg_ll(x: np.ndarray) -> float:
        return -total_ll_vec(batch, b, x[0], x[1], delta_H, delta_A)

    result = minimize(neg_ll, x0, method="L-BFGS-B",
                      bounds=[(-0.5, 0.5)] * 2,
                      options={"maxiter": 200, "ftol": 1e-8})
    return float(result.x[0]), float(result.x[1])


# ── Full sequential pipeline ─────────────────────────────────────

def run_phase1_mle(
    train_matches: list[MatchData],
    val_matches: list[MatchData],
    basis_bounds: np.ndarray,
    gamma_H_init: float = 0.0,
    gamma_A_init: float = 0.0,
    delta_H_init: np.ndarray | None = None,
    delta_A_init: np.ndarray | None = None,
    max_rounds: int = 3,
    convergence_tol: float = 0.01,
) -> Phase1MLEResult:
    """Full sequential estimation with optional re-iteration."""
    print("  Precomputing batched data...")
    train_batch = precompute_batch(train_matches, basis_bounds)
    val_batch = precompute_batch(val_matches, basis_bounds)
    print(f"  Train: {len(train_batch.iv_width)} segments, {len(train_batch.g_a)} goals")
    print(f"  Val:   {len(val_batch.iv_width)} segments, {len(val_batch.g_a)} goals")

    gamma_H = gamma_H_init
    gamma_A = gamma_A_init
    dH = delta_H_init if delta_H_init is not None else np.zeros(5)
    dA = delta_A_init if delta_A_init is not None else np.zeros(5)
    b = np.zeros(8)

    train_ll = val_ll = 0.0

    for rnd in range(1, max_rounds + 1):
        b_old = b.copy()

        # Round A: estimate b
        b = estimate_b(train_batch, gamma_H, gamma_A, dH, dA, b)
        train_ll = total_ll_vec(train_batch, b, gamma_H, gamma_A, dH, dA)
        val_ll = total_ll_vec(val_batch, b, gamma_H, gamma_A, dH, dA)
        print(f"  Round {rnd}a (b):     train_LL={train_ll:.1f}  val_LL={val_ll:.1f}")
        print(f"    b = {np.round(b, 4).tolist()}")

        # Round B: estimate delta
        dH, dA = estimate_delta(train_batch, b, gamma_H, gamma_A, dH, dA)
        train_ll = total_ll_vec(train_batch, b, gamma_H, gamma_A, dH, dA)
        val_ll = total_ll_vec(val_batch, b, gamma_H, gamma_A, dH, dA)
        print(f"  Round {rnd}b (delta): train_LL={train_ll:.1f}  val_LL={val_ll:.1f}")
        print(f"    delta_H = {np.round(dH, 4).tolist()}")
        print(f"    delta_A = {np.round(dA, 4).tolist()}")

        # Round C: estimate gamma
        gamma_H, gamma_A = estimate_gamma(train_batch, b, dH, dA)
        train_ll = total_ll_vec(train_batch, b, gamma_H, gamma_A, dH, dA)
        val_ll = total_ll_vec(val_batch, b, gamma_H, gamma_A, dH, dA)
        print(f"  Round {rnd}c (gamma): train_LL={train_ll:.1f}  val_LL={val_ll:.1f}")
        print(f"    gamma_H = {gamma_H:.4f}, gamma_A = {gamma_A:.4f}")

        if rnd > 1 and np.max(np.abs(b - b_old)) < convergence_tol:
            print(f"  Converged (b change < {convergence_tol})")
            break

    return Phase1MLEResult(
        b=b, delta_H=dH, delta_A=dA,
        gamma_H=gamma_H, gamma_A=gamma_A,
        train_LL=train_ll, val_LL=val_ll,
        n_matches_train=len(train_matches),
        n_matches_val=len(val_matches),
        rounds_completed=rnd,
    )
