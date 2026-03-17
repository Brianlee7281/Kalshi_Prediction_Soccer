"""Step 1.4 — Joint NLL Optimization (MMPP Calibration).

Jointly optimizes time profile b, red-card penalties gamma, score-difference
effects delta, and match-level baseline intensities a_H/a_A via PyTorch
gradient descent on the Poisson negative log-likelihood.

Reference: docs/phase1.md Step 1.4
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.common.types import IntervalRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_BASIS = 8  # 8 piecewise time intervals (finer resolution after 75')
_NUM_STATES = 4  # Markov red-card states
_NUM_DS_BINS = 5  # ΔS bins: ≤-2, -1, 0, +1, ≥+2

# Default optimization hyperparameters
_DEFAULT_LR = 1e-3
_DEFAULT_ADAM_EPOCHS = 1000
_DEFAULT_LAMBDA_REG = 1e-4
_DEFAULT_SIGMA_A = 0.5


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


@dataclass
class MatchData:
    """Pre-processed match data for NLL computation.

    Each match is decomposed into intervals with precomputed indices
    for basis functions, red-card states, and score-difference bins.
    """

    match_idx: int  # index into a_H / a_A arrays
    intervals: list[IntervalData] = field(default_factory=list)
    home_goal_log_lambdas: list[GoalEvent] = field(default_factory=list)
    away_goal_log_lambdas: list[GoalEvent] = field(default_factory=list)


@dataclass
class IntervalData:
    """Precomputed interval data for NLL integration term."""

    basis_idx: int  # which of the 8 basis functions
    state_X: int  # Markov state {0,1,2,3}
    ds_bin: int  # ΔS bin {0,1,2,3,4}
    duration: float  # t_end - t_start (minutes)


@dataclass
class GoalEvent:
    """Precomputed goal data for NLL point-event term."""

    basis_idx: int
    state_X: int
    ds_bin: int  # pre-goal ΔS bin


def _ds_to_bin(delta_S: int) -> int:
    """Map score difference to bin index [0, 4]."""
    if delta_S <= -2:
        return 0
    if delta_S >= 2:
        return 4
    return delta_S + 2


def _time_to_basis(t: float, alpha_1: float = 0.0) -> int:
    """Map match minute to basis function index [0, 7].

    Basis boundaries (with stoppage):
        0: [0, 15)          — early first half
        1: [15, 30)         — mid first half
        2: [30, 45+α₁)      — late first half + stoppage
        3: [45+α₁, 60+α₁)   — early second half
        4: [60+α₁, 75+α₁)   — mid second half
        5: [75+α₁, 85+α₁)   — late second half
        6: [85+α₁, 90+α₁)   — pre-stoppage
        7: [90+α₁, T_m)     — stoppage time
    """
    ht = 45.0 + alpha_1
    if t < 15.0:
        return 0
    if t < 30.0:
        return 1
    if t < ht:
        return 2
    if t < ht + 15.0:
        return 3
    if t < ht + 30.0:
        return 4
    if t < ht + 40.0:
        return 5
    if t < ht + 45.0:
        return 6
    return 7


def prepare_match_data(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
) -> list[MatchData]:
    """Convert interval records into MatchData for NLL computation.

    Args:
        intervals_by_match: Dict mapping match_id to its intervals.
        match_ids: Ordered list of match IDs (index = match_idx).

    Returns:
        List of MatchData, one per match.
    """
    match_id_to_idx = {mid: i for i, mid in enumerate(match_ids)}
    result: list[MatchData] = []

    for mid in match_ids:
        ivs = intervals_by_match.get(mid, [])
        md = MatchData(match_idx=match_id_to_idx[mid])

        for iv in ivs:
            if iv.is_halftime:
                continue

            duration = iv.t_end - iv.t_start
            if duration <= 0:
                continue

            alpha_1 = iv.alpha_1
            basis_idx = _time_to_basis(iv.t_start, alpha_1)

            md.intervals.append(IntervalData(
                basis_idx=basis_idx,
                state_X=iv.state_X,
                ds_bin=_ds_to_bin(iv.delta_S),
                duration=duration,
            ))

            # Point events: home goals
            for i, t_g in enumerate(iv.home_goal_times):
                pre_ds = iv.goal_delta_before[i] if i < len(iv.goal_delta_before) else iv.delta_S
                md.home_goal_log_lambdas.append(GoalEvent(
                    basis_idx=_time_to_basis(t_g, alpha_1),
                    state_X=iv.state_X,
                    ds_bin=_ds_to_bin(pre_ds),
                ))

            # Point events: away goals
            n_home = len(iv.home_goal_times)
            for j, t_g in enumerate(iv.away_goal_times):
                idx = n_home + j
                pre_ds = iv.goal_delta_before[idx] if idx < len(iv.goal_delta_before) else iv.delta_S
                md.away_goal_log_lambdas.append(GoalEvent(
                    basis_idx=_time_to_basis(t_g, alpha_1),
                    state_X=iv.state_X,
                    ds_bin=_ds_to_bin(pre_ds),
                ))

        result.append(md)

    return result


# ---------------------------------------------------------------------------
# Parametric delta function
# ---------------------------------------------------------------------------


def parametric_delta(
    s: torch.Tensor,
    beta: torch.Tensor,
    kappa: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Compute δ(s) = β·s + κ·sign(s)·(1 - exp(-|s|/τ)).

    Args:
        s: Score difference values (tensor).
        beta: Linear slope parameter.
        kappa: Saturation magnitude parameter.
        tau: Saturation rate parameter (positive).

    Returns:
        Delta values (same shape as s).
    """
    abs_s = torch.abs(s)
    sign_s = torch.sign(s)
    return beta * s + kappa * sign_s * (1.0 - torch.exp(-abs_s / tau))


def delta_lookup_from_params(
    beta: float,
    kappa: float,
    tau: float,
) -> np.ndarray:
    """Generate the 5-element δ lookup table from parametric coefficients.

    Returns:
        Array of shape (5,) with δ values for ΔS = {≤-2, -1, 0, +1, ≥+2}.
    """
    s_vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    delta = parametric_delta(
        s_vals,
        torch.tensor(beta),
        torch.tensor(kappa),
        torch.tensor(tau),
    )
    return delta.detach().numpy()


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------


class MMPPModel(nn.Module):
    """MMPP parameter model for joint NLL optimization.

    Learnable parameters:
        a_H, a_A: (M,) match-level home/away baseline intensities
        b: (8,) time-profile basis coefficients
        gamma_H_raw: (2,) home red-card penalties [γ^H_1, γ^H_2]
        gamma_A_raw: (2,) away red-card penalties [γ^A_1, γ^A_2]
        beta_H, kappa_H, log_tau_H: home parametric delta coefficients
        beta_A, kappa_A, log_tau_A: away parametric delta coefficients
    """

    def __init__(
        self,
        n_matches: int,
        a_H_init: torch.Tensor,
        a_A_init: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_matches = n_matches

        # Match-level baselines (initialized from XGBoost prior)
        self.a_H = nn.Parameter(a_H_init.clone())
        self.a_A = nn.Parameter(a_A_init.clone())

        # Store initial values for regularization
        self.register_buffer("a_H_init", a_H_init.clone())
        self.register_buffer("a_A_init", a_A_init.clone())

        # Time profile: 8 basis coefficients, init = 0
        self.b = nn.Parameter(torch.zeros(_NUM_BASIS))

        # Red-card gamma (free parameters: 2 per team)
        # γ^H = [0, γ^H_1, γ^H_2, γ^H_1 + γ^H_2]
        self.gamma_H_raw = nn.Parameter(torch.tensor([-0.05, 0.05]))
        self.gamma_A_raw = nn.Parameter(torch.tensor([0.05, -0.05]))

        # Parametric delta coefficients (per team)
        self.beta_H = nn.Parameter(torch.tensor(0.0))
        self.kappa_H = nn.Parameter(torch.tensor(0.0))
        self.log_tau_H = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0

        self.beta_A = nn.Parameter(torch.tensor(0.0))
        self.kappa_A = nn.Parameter(torch.tensor(0.0))
        self.log_tau_A = nn.Parameter(torch.tensor(0.0))

    def get_gamma_H(self) -> torch.Tensor:
        """Build 4-element γ^H vector with clamping.

        Returns:
            Tensor of shape (4,): [0, γ^H_1, γ^H_2, γ^H_1 + γ^H_2]
        """
        g1 = self.gamma_H_raw[0].clamp(-1.5, 0.0)
        g2 = self.gamma_H_raw[1].clamp(0.0, 1.5)
        return torch.stack([torch.tensor(0.0), g1, g2, g1 + g2])

    def get_gamma_A(self) -> torch.Tensor:
        """Build 4-element γ^A vector with clamping.

        Returns:
            Tensor of shape (4,): [0, γ^A_1, γ^A_2, γ^A_1 + γ^A_2]
        """
        g1 = self.gamma_A_raw[0].clamp(0.0, 1.5)
        g2 = self.gamma_A_raw[1].clamp(-1.5, 0.0)
        return torch.stack([torch.tensor(0.0), g1, g2, g1 + g2])

    def get_b_clamped(self) -> torch.Tensor:
        """Return b with clamping to [-0.5, 0.5]."""
        return self.b.clamp(-0.5, 0.5)

    def get_tau_H(self) -> torch.Tensor:
        """Return τ_H = exp(log_τ_H), clamped to [0.1, 5.0]."""
        return torch.exp(self.log_tau_H).clamp(0.1, 5.0)

    def get_tau_A(self) -> torch.Tensor:
        """Return τ_A = exp(log_τ_A), clamped to [0.1, 5.0]."""
        return torch.exp(self.log_tau_A).clamp(0.1, 5.0)

    def get_beta_H_clamped(self) -> torch.Tensor:
        return self.beta_H.clamp(-0.5, 0.5)

    def get_beta_A_clamped(self) -> torch.Tensor:
        return self.beta_A.clamp(-0.5, 0.5)

    def get_kappa_H_clamped(self) -> torch.Tensor:
        return self.kappa_H.clamp(-1.0, 1.0)

    def get_kappa_A_clamped(self) -> torch.Tensor:
        return self.kappa_A.clamp(-1.0, 1.0)

    def get_delta_H(self) -> torch.Tensor:
        """Compute 5-element δ_H lookup [ΔS = -2, -1, 0, +1, +2]."""
        s = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        return parametric_delta(
            s, self.get_beta_H_clamped(), self.get_kappa_H_clamped(), self.get_tau_H(),
        )

    def get_delta_A(self) -> torch.Tensor:
        """Compute 5-element δ_A lookup [ΔS = -2, -1, 0, +1, +2]."""
        s = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        return parametric_delta(
            s, self.get_beta_A_clamped(), self.get_kappa_A_clamped(), self.get_tau_A(),
        )

    def log_lambda_H(
        self,
        match_idx: int,
        basis_idx: int,
        state_X: int,
        ds_bin: int,
    ) -> torch.Tensor:
        """Compute ln λ_H for a given interval/event context."""
        b = self.get_b_clamped()
        gamma_H = self.get_gamma_H()
        delta_H = self.get_delta_H()
        return self.a_H[match_idx] + b[basis_idx] + gamma_H[state_X] + delta_H[ds_bin]

    def log_lambda_A(
        self,
        match_idx: int,
        basis_idx: int,
        state_X: int,
        ds_bin: int,
    ) -> torch.Tensor:
        """Compute ln λ_A for a given interval/event context."""
        b = self.get_b_clamped()
        gamma_A = self.get_gamma_A()
        delta_A = self.get_delta_A()
        return self.a_A[match_idx] + b[basis_idx] + gamma_A[state_X] + delta_A[ds_bin]


# ---------------------------------------------------------------------------
# NLL computation
# ---------------------------------------------------------------------------


def compute_nll(
    model: MMPPModel,
    match_data: list[MatchData],
    *,
    sigma_a: float = _DEFAULT_SIGMA_A,
    lambda_reg: float = _DEFAULT_LAMBDA_REG,
) -> torch.Tensor:
    """Compute the full NLL loss over all matches.

    Loss = -Σ_m [Σ_g ln λ_H(t_g) + Σ_g ln λ_A(t_g) - Σ_k (μ^H_k + μ^A_k)]
           + (1/2σ²_a) Σ_m [(a^m_H - a^m_init_H)² + (a^m_A - a^m_init_A)²]
           + λ_reg (||b||² + ||γ^H||² + ||γ^A||² + ||δ_H||² + ||δ_A||²)

    Args:
        model: MMPPModel with current parameters.
        match_data: Pre-processed match data.
        sigma_a: Standard deviation for ML prior regularization.
        lambda_reg: L2 regularization coefficient.

    Returns:
        Scalar loss tensor.
    """
    # Precompute clamped parameters once
    b = model.get_b_clamped()
    gamma_H = model.get_gamma_H()
    gamma_A = model.get_gamma_A()
    delta_H = model.get_delta_H()
    delta_A = model.get_delta_A()

    nll = torch.tensor(0.0)

    for md in match_data:
        m = md.match_idx

        # --- Integration term: Σ_k (μ^H_k + μ^A_k) ---
        for iv in md.intervals:
            log_lam_H = model.a_H[m] + b[iv.basis_idx] + gamma_H[iv.state_X] + delta_H[iv.ds_bin]
            log_lam_A = model.a_A[m] + b[iv.basis_idx] + gamma_A[iv.state_X] + delta_A[iv.ds_bin]
            mu_H = torch.exp(log_lam_H) * iv.duration
            mu_A = torch.exp(log_lam_A) * iv.duration
            nll = nll + mu_H + mu_A

        # --- Point-event term: -Σ_g ln λ ---
        for ge in md.home_goal_log_lambdas:
            log_lam = model.a_H[m] + b[ge.basis_idx] + gamma_H[ge.state_X] + delta_H[ge.ds_bin]
            nll = nll - log_lam

        for ge in md.away_goal_log_lambdas:
            log_lam = model.a_A[m] + b[ge.basis_idx] + gamma_A[ge.state_X] + delta_A[ge.ds_bin]
            nll = nll - log_lam

    # --- ML prior regularization: (1/2σ²_a) Σ_m [(a_H - a_H_init)² + ...] ---
    if sigma_a > 0:
        inv_2sigma2 = 1.0 / (2.0 * sigma_a * sigma_a)
        a_H_init: torch.Tensor = model.a_H_init  # type: ignore[assignment]
        a_A_init: torch.Tensor = model.a_A_init  # type: ignore[assignment]
        reg_a = inv_2sigma2 * (
            torch.sum((model.a_H - a_H_init) ** 2)
            + torch.sum((model.a_A - a_A_init) ** 2)
        )
        nll = nll + reg_a

    # --- L2 regularization on structural parameters ---
    reg_l2 = lambda_reg * (
        torch.sum(b ** 2)
        + torch.sum(gamma_H ** 2)
        + torch.sum(gamma_A ** 2)
        + torch.sum(delta_H ** 2)
        + torch.sum(delta_A ** 2)
    )
    nll = nll + reg_l2

    return nll


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Result of NLL optimization."""

    b: np.ndarray  # (8,) time profile
    gamma_H: np.ndarray  # (4,) home red-card penalties
    gamma_A: np.ndarray  # (4,) away red-card penalties
    delta_H: np.ndarray  # (5,) home score-difference lookup
    delta_A: np.ndarray  # (5,) away score-difference lookup
    a_H: np.ndarray  # (M,) corrected home baselines
    a_A: np.ndarray  # (M,) corrected away baselines
    beta_H: float
    kappa_H: float
    tau_H: float
    beta_A: float
    kappa_A: float
    tau_A: float
    loss_history: list[float] = field(default_factory=list)


def optimize_nll(
    match_data: list[MatchData],
    a_H_init: np.ndarray,
    a_A_init: np.ndarray,
    *,
    sigma_a: float = _DEFAULT_SIGMA_A,
    lambda_reg: float = _DEFAULT_LAMBDA_REG,
    lr: float = _DEFAULT_LR,
    num_epochs: int = _DEFAULT_ADAM_EPOCHS,
) -> OptimizationResult:
    """Run Adam optimization on the joint NLL.

    Args:
        match_data: Pre-processed match data from prepare_match_data.
        a_H_init: Initial home log-intensities from XGBoost (shape M).
        a_A_init: Initial away log-intensities from XGBoost (shape M).
        sigma_a: ML prior regularization strength.
        lambda_reg: L2 regularization coefficient.
        lr: Adam learning rate.
        num_epochs: Number of Adam epochs.

    Returns:
        OptimizationResult with all fitted parameters and loss history.
    """
    n_matches = len(a_H_init)
    model = MMPPModel(
        n_matches=n_matches,
        a_H_init=torch.tensor(a_H_init, dtype=torch.float32),
        a_A_init=torch.tensor(a_A_init, dtype=torch.float32),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    for _epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_nll(model, match_data, sigma_a=sigma_a, lambda_reg=lambda_reg)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        loss_history.append(float(loss.item()))

    return _extract_result(model, loss_history)


def _extract_result(model: MMPPModel, loss_history: list[float]) -> OptimizationResult:
    """Extract numpy arrays from trained model."""
    with torch.no_grad():
        return OptimizationResult(
            b=model.get_b_clamped().numpy(),
            gamma_H=model.get_gamma_H().numpy(),
            gamma_A=model.get_gamma_A().numpy(),
            delta_H=model.get_delta_H().numpy(),
            delta_A=model.get_delta_A().numpy(),
            a_H=model.a_H.numpy(),
            a_A=model.a_A.numpy(),
            beta_H=float(model.get_beta_H_clamped()),
            kappa_H=float(model.get_kappa_H_clamped()),
            tau_H=float(model.get_tau_H()),
            beta_A=float(model.get_beta_A_clamped()),
            kappa_A=float(model.get_kappa_A_clamped()),
            tau_A=float(model.get_tau_A()),
            loss_history=loss_history,
        )
