"""Extended Kalman Filter for live team strength estimation (v5).

Tracks a_H(t) and a_A(t) with uncertainty P_H(t), P_A(t).
Replaces the Bayesian shrinkage in InPlayStrengthUpdater.

EKF equations (from docs/MMPP_v5_Complete.md §7.1):
  Prediction:   P_i(t|t-dt) = P_i(t-dt|t-dt) + σ²_ω × dt
  Goal update:  K = P_i / (P_i·λ_i + 1); a_i += K·(1 - λ_i·dt); P_i = (1-K·λ_i)·P_i
  No-goal:      K₀ = P_i·λ_i / (P_i·λ_i + 1); a_i += K₀·(0 - λ_i·dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EKFStrengthTracker:
    """Extended Kalman Filter for live team strength.

    State: a_H(t), a_A(t) — current log-intensity estimates
    Uncertainty: P_H(t), P_A(t) — estimate variance
    """

    # Current estimates (mutable)
    a_H: float
    a_A: float
    P_H: float  # uncertainty for home
    P_A: float  # uncertainty for away

    # Process noise (immutable after init)
    sigma_omega_sq: float

    # Initial values (immutable, for clamping)
    _a_H_init: float = 0.0
    _a_A_init: float = 0.0

    def __init__(
        self,
        a_H_init: float,
        a_A_init: float,
        P_0: float,
        sigma_omega_sq: float = 0.01,
    ) -> None:
        self.a_H = a_H_init
        self.a_A = a_A_init
        self.P_H = P_0
        self.P_A = P_0
        self.sigma_omega_sq = sigma_omega_sq
        self._a_H_init = a_H_init
        self._a_A_init = a_A_init

    def predict(self, dt: float) -> None:
        """Prediction step: uncertainty grows by σ²_ω × dt.

        Called once per tick (dt=1/60 minutes or dt=1.0 seconds depending on convention).
        """
        self.P_H += self.sigma_omega_sq * dt
        self.P_A += self.sigma_omega_sq * dt

    def update_goal(
        self, team: str, lambda_H: float, lambda_A: float, dt: float = 1.0,
    ) -> None:
        """Update on goal scored.

        K = P / (P·λ + 1)
        a += K · (1 - λ·dt)
        P = (1 - K·λ) · P
        """
        if team == "home":
            K = self.P_H / (self.P_H * lambda_H + 1.0) if lambda_H > 0 else 0.0
            self.a_H += K * (1.0 - lambda_H * dt)
            self.P_H = (1.0 - K * lambda_H) * self.P_H
        else:
            K = self.P_A / (self.P_A * lambda_A + 1.0) if lambda_A > 0 else 0.0
            self.a_A += K * (1.0 - lambda_A * dt)
            self.P_A = (1.0 - K * lambda_A) * self.P_A

        # Clamp to prevent divergence (±1.5 from initial)
        self.a_H = max(self._a_H_init - 1.5, min(self._a_H_init + 1.5, self.a_H))
        self.a_A = max(self._a_A_init - 1.5, min(self._a_A_init + 1.5, self.a_A))

    def update_no_goal(
        self, lambda_H: float, lambda_A: float, dt: float = 1.0,
    ) -> None:
        """Update on no-goal observation (weak negative evidence).

        K₀ = P·λ / (P·λ + 1)
        innovation = 0 - λ·dt  (expected goals in dt, observed 0)
        a += K₀ · innovation
        """
        if lambda_H > 0:
            K0_H = self.P_H * lambda_H / (self.P_H * lambda_H + 1.0)
            self.a_H += K0_H * (0.0 - lambda_H * dt)
        if lambda_A > 0:
            K0_A = self.P_A * lambda_A / (self.P_A * lambda_A + 1.0)
            self.a_A += K0_A * (0.0 - lambda_A * dt)

        # Clamp
        self.a_H = max(self._a_H_init - 1.5, min(self._a_H_init + 1.5, self.a_H))
        self.a_A = max(self._a_A_init - 1.5, min(self._a_A_init + 1.5, self.a_A))

    def compute_surprise_score(
        self,
        team: str,
        P_model_home_win: float,
        P_model_away_win: float | None = None,
    ) -> float:
        """Compute SurpriseScore = 1 - P(scoring team wins | pre-goal state).

        For away goals, pass *P_model_away_win* explicitly to avoid
        conflating P(A) with 1 - P(H) = P(A) + P(D).  When omitted,
        falls back to the old approximation ``1 - P_model_home_win``.

        Returns float in [0, 1]. Higher = more surprising.
        """
        if team == "home":
            scoring_team_win_prob = P_model_home_win
        else:
            if P_model_away_win is not None:
                scoring_team_win_prob = P_model_away_win
            else:
                scoring_team_win_prob = 1.0 - P_model_home_win
        return max(0.0, min(1.0, 1.0 - scoring_team_win_prob))

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Returns (a_H, a_A, P_H, P_A)."""
        return self.a_H, self.a_A, self.P_H, self.P_A
