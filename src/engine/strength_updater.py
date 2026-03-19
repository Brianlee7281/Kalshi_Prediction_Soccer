"""InPlayStrengthUpdater — v5 EKF-based team strength updates.

Delegates to EKFStrengthTracker for Kalman filter updates.
Provides backward-compatible classify_goal() that maps SurpriseScore
to categorical labels (SURPRISE/EXPECTED/NEUTRAL).

v4 used Bayesian shrinkage. v5 uses Extended Kalman Filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.engine.ekf import EKFStrengthTracker


@dataclass
class GoalClassification:
    """Result of classifying a goal event."""
    label: str  # "SURPRISE" | "EXPECTED" | "NEUTRAL"
    team: str   # "home" | "away"
    scoring_team_prob: float  # pre-match win probability of scoring team


@dataclass
class StrengthSnapshot:
    """Snapshot of updated strengths after a goal, for logging."""
    a_H: float
    a_A: float
    a_H_init: float
    a_A_init: float
    n_H: int
    n_A: int
    shrink_H: float
    shrink_A: float
    classification: GoalClassification


class InPlayStrengthUpdater:
    """v5: EKF-based updater for team log-intensities during a live match."""

    def __init__(
        self,
        a_H_init: float,
        a_A_init: float,
        sigma_a_sq: float,
        pre_match_home_prob: float,
        ekf_tracker: EKFStrengthTracker | None = None,
    ) -> None:
        self.a_H_init = a_H_init
        self.a_A_init = a_A_init
        self.sigma_a_sq = sigma_a_sq
        self.pre_match_home_prob = pre_match_home_prob

        # v5: delegate to EKF
        self.ekf = ekf_tracker or EKFStrengthTracker(
            a_H_init=a_H_init,
            a_A_init=a_A_init,
            P_0=sigma_a_sq,
            sigma_omega_sq=0.01,
        )

        # Current values (mirror EKF state)
        self.a_H = a_H_init
        self.a_A = a_A_init
        self.n_H = 0
        self.n_A = 0

    def update_on_goal(
        self,
        team: str,
        mu_H_elapsed: float = 0.0,
        mu_A_elapsed: float = 0.0,
        *,
        lambda_H: float = 0.0,
        lambda_A: float = 0.0,
        dt: float = 1.0,
    ) -> tuple[float, float]:
        """Update a_H and a_A after a goal event.

        v5 path: uses lambda_H/lambda_A kwargs for EKF update.
        v4 compat: if lambda_H/lambda_A not provided, uses Bayesian fallback.
        """
        if team == "home":
            self.n_H += 1
        else:
            self.n_A += 1

        if lambda_H > 0 or lambda_A > 0:
            # v5 EKF path
            self.ekf.update_goal(team, lambda_H, lambda_A, dt)
            self.a_H = self.ekf.a_H
            self.a_A = self.ekf.a_A
        else:
            # v4 Bayesian fallback
            self.a_H = self._bayesian_update(self.a_H_init, self.n_H, mu_H_elapsed)
            self.a_A = self._bayesian_update(self.a_A_init, self.n_A, mu_A_elapsed)
            self.ekf.a_H = self.a_H
            self.ekf.a_A = self.a_A

        return self.a_H, self.a_A

    def predict(self, dt: float) -> None:
        """EKF prediction step."""
        self.ekf.predict(dt)
        self.a_H = self.ekf.a_H
        self.a_A = self.ekf.a_A

    def update_no_goal(self, lambda_H: float, lambda_A: float, dt: float) -> None:
        """EKF no-goal update."""
        self.ekf.update_no_goal(lambda_H, lambda_A, dt)
        self.a_H = self.ekf.a_H
        self.a_A = self.ekf.a_A

    def compute_surprise_score(
        self,
        team: str,
        P_model_home_win: float,
        P_model_away_win: float | None = None,
    ) -> float:
        """SurpriseScore = 1 - P(scoring team wins)."""
        return self.ekf.compute_surprise_score(team, P_model_home_win, P_model_away_win)

    def classify_goal(self, team: str) -> GoalClassification:
        """Classify goal using pre_match_home_prob thresholds.

        Backward compatible with v4 categorical labels.
        """
        if team == "home":
            scoring_prob = self.pre_match_home_prob
        else:
            scoring_prob = 1.0 - self.pre_match_home_prob

        if scoring_prob < 0.35:
            label = "SURPRISE"
        elif scoring_prob > 0.60:
            label = "EXPECTED"
        else:
            label = "NEUTRAL"

        return GoalClassification(
            label=label, team=team, scoring_team_prob=scoring_prob,
        )

    def snapshot(self, classification: GoalClassification) -> StrengthSnapshot:
        return StrengthSnapshot(
            a_H=self.a_H, a_A=self.a_A,
            a_H_init=self.a_H_init, a_A_init=self.a_A_init,
            n_H=self.n_H, n_A=self.n_A,
            shrink_H=self._shrink_factor(self.n_H * 1.0),
            shrink_A=self._shrink_factor(self.n_A * 1.0),
            classification=classification,
        )

    def _bayesian_update(self, a_prior: float, n_actual: int, mu_elapsed: float) -> float:
        """v4 Bayesian fallback."""
        if mu_elapsed <= 0.0:
            return a_prior
        shrink = mu_elapsed / (mu_elapsed + self.sigma_a_sq)
        correction = math.log((n_actual + 0.5) / (mu_elapsed + 0.5))
        correction = max(-0.3, min(0.3, correction))
        return a_prior + shrink * correction

    def _shrink_factor(self, mu_elapsed: float) -> float:
        if mu_elapsed <= 0.0:
            return 0.0
        return mu_elapsed / (mu_elapsed + self.sigma_a_sq)
