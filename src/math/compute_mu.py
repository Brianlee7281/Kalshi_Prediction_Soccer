"""Step 3.2: Remaining Expected Goals Calculation.

Computes μ_H(t, T) and μ_A(t, T) — the expected number of goals remaining
from current time t to match end T, using the MMPP integral formula.

Formula (with team-specific gamma):
  μ_H(t, T) = Σ_ℓ Σ_j P̄[X(t),j]^(ℓ) · exp(a_H + b[i_ℓ] + γ^H_j + δ_H[di]) · Δτ_ℓ
  μ_A(t, T) = Σ_ℓ Σ_j P̄[X(t),j]^(ℓ) · exp(a_A + b[i_ℓ] + γ^A_j + δ_A[di]) · Δτ_ℓ

Where:
  P̄[X(t),j]^(ℓ) = transition probability from X(t) to j over the time
                   elapsed to the start of subinterval ℓ (approximation:
                   midpoint quadrature using P_grid)
  i_ℓ            = basis index for subinterval ℓ
  di             = delta index = clamp(ΔS + 2, 0, 4)
  Δτ_ℓ           = subinterval length in minutes

Reference: docs/phase3.md §Step 3.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel as LiveFootballQuantModel

logger = get_logger("compute_mu")

# Fine-grid threshold (minutes remaining): use P_fine_grid below this
_FINE_GRID_THRESHOLD = 5.0


def get_transition_prob(model: LiveFootballQuantModel, dt_min: float) -> np.ndarray:
    """Look up transition probability matrix P(dt) from precomputed grid.

    Uses fine grid (10-second increments) when close to match end,
    standard grid (1-minute increments) otherwise.

    Args:
        model: Live model with precomputed P_grid / P_fine_grid.
        dt_min: Duration in minutes for the transition.

    Returns:
        Transition probability matrix of shape (4, 4).
    """
    if dt_min <= _FINE_GRID_THRESHOLD and model.P_fine_grid:
        # Fine grid: steps are 1/6 minute (10 seconds)
        dt_10sec = int(round(dt_min * 6))
        dt_10sec = max(0, min(30, dt_10sec))
        p = model.P_fine_grid.get(dt_10sec)
        if p is not None:
            return p

    # Standard grid: 1-minute increments, capped at 100
    dt_round = max(0, min(100, round(dt_min)))
    p_std = model.P_grid.get(dt_round)
    if p_std is not None:
        return p_std

    # Fallback: identity (no transition)
    return np.eye(4, dtype=np.float64)


def compute_remaining_mu(
    model: LiveFootballQuantModel,
    override_delta_S: int | None = None,
) -> tuple[float, float]:
    """Compute remaining expected goals μ_H and μ_A for the current tick.

    Splits [t, T_exp] at basis_bounds into subintervals and sums weighted
    intensity contributions using the Markov transition probability grid.

    Midpoint quadrature: for each subinterval, the transition probability
    P̄[X,j] is evaluated at the midpoint of the interval relative to t.

    Args:
        model: Current match state.
        override_delta_S: Optional override for ΔS (used in preliminary
            precomputation before score is committed).

    Returns:
        (mu_H, mu_A) — remaining expected goals, non-negative floats.
    """
    t = model.t
    T = model.T_exp
    X = model.current_state_X
    ds = override_delta_S if override_delta_S is not None else model.delta_S

    # delta index: ΔS → {0: ≤-2, 1: -1, 2: 0, 3: +1, 4: ≥+2}
    # Home uses home perspective (ds + 2), away uses away perspective (-ds + 2)
    # to match MLE calibration in phase1_mle.py _add_segments
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))

    if t >= T:
        return 0.0, 0.0

    # Build list of subinterval boundaries clipped to [t, T]
    boundaries = [float(b) for b in model.basis_bounds if t < float(b) < T]
    breakpoints = [t] + boundaries + [T]

    mu_H = 0.0
    mu_A = 0.0

    for idx in range(len(breakpoints) - 1):
        tau_start = breakpoints[idx]
        tau_end = breakpoints[idx + 1]
        delta_tau = tau_end - tau_start

        if delta_tau <= 0.0:
            continue

        # Basis index: which interval does tau_start fall in?
        n_periods = len(model.basis_bounds) - 1
        bi = 0
        for k in range(n_periods):
            b_lo = float(model.basis_bounds[k])
            b_hi = float(model.basis_bounds[k + 1])
            if b_lo <= tau_start < b_hi:
                bi = k
                break

        # Midpoint quadrature: evaluate P at midpoint of subinterval
        dt_mid = ((tau_start + tau_end) / 2.0) - t
        P_mid = get_transition_prob(model, dt_mid)
        p_row = P_mid[X, :]  # shape (4,): P(X -> j) at midpoint

        # Intensity weighted by state-occupancy probabilities
        b_val = float(model.b[bi])

        contrib_H = 0.0
        contrib_A = 0.0
        for j in range(4):
            p_j = float(p_row[j])
            if p_j > 0.0:
                contrib_H += p_j * np.exp(
                    model.a_H + b_val + float(model.gamma_H[j]) + float(model.delta_H[di_H])
                )
                contrib_A += p_j * np.exp(
                    model.a_A + b_val + float(model.gamma_A[j]) + float(model.delta_A[di_A])
                )

        mu_H += contrib_H * delta_tau
        mu_A += contrib_A * delta_tau

    return max(0.0, mu_H), max(0.0, mu_A)


def compute_remaining_mu_v5(
    model: LiveFootballQuantModel,
    override_delta_S: int | None = None,
) -> tuple[float, float]:
    """Compute remaining expected goals with v5 asymmetric delta and eta stoppage.

    If the model has ``delta_H_pos`` attribute (v5 params loaded), uses
    asymmetric delta lookup based on sign of delta_S and applies eta
    multipliers during stoppage subintervals. Otherwise falls back to
    the original symmetric logic via :func:`compute_remaining_mu`.

    Args:
        model: Current match state (may or may not carry v5 attributes).
        override_delta_S: Optional override for ΔS.

    Returns:
        (mu_H, mu_A) — remaining expected goals, non-negative floats.
    """
    # Fall back to v4 if model lacks v5 params
    if not hasattr(model, "delta_H_pos") or model.delta_H_pos is None:
        return compute_remaining_mu(model, override_delta_S)

    t = model.t
    T = model.T_exp
    X = model.current_state_X
    ds = override_delta_S if override_delta_S is not None else model.delta_S

    # delta index: each team uses its own perspective to match MLE calibration
    # Home uses home perspective (ds + 2), away uses away perspective (-ds + 2)
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))

    if t >= T:
        return 0.0, 0.0

    # Asymmetric delta lookup
    if ds > 0:  # home leading
        dH_val = float(model.delta_H_pos[di_H])
        dA_val = float(model.delta_A_pos[di_A])
    else:  # trailing or tied
        dH_val = float(model.delta_H_neg[di_H])
        dA_val = float(model.delta_A_neg[di_A])

    # Stoppage parameters
    eta_H = getattr(model, "eta_H", 0.0)
    eta_A = getattr(model, "eta_A", 0.0)
    eta_H2 = getattr(model, "eta_H2", 0.0)
    eta_A2 = getattr(model, "eta_A2", 0.0)
    stoppage_1_start = getattr(model, "stoppage_1_start", 45.0)
    stoppage_2_start = getattr(model, "stoppage_2_start", 90.0)

    # Build list of subinterval boundaries clipped to [t, T]
    boundaries = [float(b) for b in model.basis_bounds if t < float(b) < T]
    breakpoints = [t] + boundaries + [T]

    mu_H = 0.0
    mu_A = 0.0

    for idx in range(len(breakpoints) - 1):
        tau_start = breakpoints[idx]
        tau_end = breakpoints[idx + 1]
        delta_tau = tau_end - tau_start

        if delta_tau <= 0.0:
            continue

        # Basis index: which interval does tau_start fall in?
        n_periods = len(model.basis_bounds) - 1
        bi = 0
        for k in range(n_periods):
            b_lo = float(model.basis_bounds[k])
            b_hi = float(model.basis_bounds[k + 1])
            if b_lo <= tau_start < b_hi:
                bi = k
                break

        # Eta stoppage multiplier at subinterval midpoint
        tau_mid = (tau_start + tau_end) / 2.0
        eta_h = 0.0
        eta_a = 0.0
        if stoppage_1_start < tau_mid < stoppage_1_start + 10.0:
            eta_h = eta_H
            eta_a = eta_A
        elif stoppage_2_start < tau_mid < stoppage_2_start + 10.0:
            eta_h = eta_H2
            eta_a = eta_A2

        # Midpoint quadrature: evaluate P at midpoint of subinterval
        dt_mid = tau_mid - t
        P_mid = get_transition_prob(model, dt_mid)
        p_row = P_mid[X, :]  # shape (4,): P(X -> j) at midpoint

        # Intensity weighted by state-occupancy probabilities
        b_val = float(model.b[bi])

        contrib_H = 0.0
        contrib_A = 0.0
        for j in range(4):
            p_j = float(p_row[j])
            if p_j > 0.0:
                contrib_H += p_j * np.exp(
                    model.a_H + b_val + float(model.gamma_H[j]) + dH_val + eta_h
                )
                contrib_A += p_j * np.exp(
                    model.a_A + b_val + float(model.gamma_A[j]) + dA_val + eta_a
                )

        mu_H += contrib_H * delta_tau
        mu_A += contrib_A * delta_tau

    return max(0.0, mu_H), max(0.0, mu_A)
