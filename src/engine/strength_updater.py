"""InPlayStrengthUpdater — Bayesian update of a_H/a_A on goal events.

Formula: empirical Bayes shrinkage for Poisson rates (normal-normal approximation).
  shrink     = mu_elapsed / (mu_elapsed + sigma_a_sq)
  correction = log((n_actual + 0.5) / (mu_elapsed + 0.5))  [Laplace smoothing]
  a_new      = a_prior + shrink * correction
Inspired by state-space filtering concepts (Kalman gain analogy) but implemented
as a closed-form approximation suitable for real-time in-play updating.
NOT the SPDK method from Koopman & Lit (2015) — that paper updates week-to-week
using importance sampling, not in-play.

sigma_a is reused from production_params (Phase 1). No additional
training required.

Reference: docs/architecture.md §3.3 item 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GoalClassification:
    """Result of classifying a goal event."""

    label: str  # "SURPRISE" | "EXPECTED" | "NEUTRAL"
    team: str  # "home" | "away"
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
    """Bayesian updater for team log-intensities during a live match.

    Maintains running goal counts and computes shrinkage-adjusted a_H/a_A
    after each goal event. Preserves initial values for delta display.
    """

    def __init__(
        self,
        a_H_init: float,
        a_A_init: float,
        sigma_a_sq: float,
        pre_match_home_prob: float,
    ) -> None:
        """Initialize the updater.

        Args:
            a_H_init: Initial home log-intensity from Phase 2 backsolve.
            a_A_init: Initial away log-intensity from Phase 2 backsolve.
            sigma_a_sq: Variance of the ML prior (sigma_a^2 from Phase 1).
            pre_match_home_prob: Pre-match home win probability (vig-removed).
        """
        # Immutable originals
        self.a_H_init: float = a_H_init
        self.a_A_init: float = a_A_init
        self.sigma_a_sq: float = sigma_a_sq
        self.pre_match_home_prob: float = pre_match_home_prob

        # Mutable current values
        self.a_H: float = a_H_init
        self.a_A: float = a_A_init

        # Running goal counts
        self.n_H: int = 0
        self.n_A: int = 0

    def update_on_goal(
        self,
        team: str,
        mu_H_elapsed: float,
        mu_A_elapsed: float,
    ) -> tuple[float, float]:
        """Update a_H and a_A after a goal event.

        Args:
            team: Which team scored — "home" or "away".
            mu_H_elapsed: Expected home goals elapsed so far
                (mu_H_at_kickoff - mu_H_current).
            mu_A_elapsed: Expected away goals elapsed so far
                (mu_A_at_kickoff - mu_A_current).

        Returns:
            (new_a_H, new_a_A) — updated log-intensities.
        """
        if team == "home":
            self.n_H += 1
        else:
            self.n_A += 1

        self.a_H = self._bayesian_update(self.a_H_init, self.n_H, mu_H_elapsed)
        self.a_A = self._bayesian_update(self.a_A_init, self.n_A, mu_A_elapsed)

        return self.a_H, self.a_A

    def classify_goal(
        self,
        team: str,
    ) -> GoalClassification:
        """Classify a goal as SURPRISE, EXPECTED, or NEUTRAL.

        Based on the pre-match win probability of the scoring team:
          SURPRISE:  scoring team prob < 0.35
          EXPECTED:  scoring team prob > 0.60
          NEUTRAL:   otherwise

        Args:
            team: Which team scored — "home" or "away".

        Returns:
            GoalClassification with label and metadata.
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
            label=label,
            team=team,
            scoring_team_prob=scoring_prob,
        )

    def snapshot(self, classification: GoalClassification) -> StrengthSnapshot:
        """Create a snapshot of the current state for logging.

        Args:
            classification: The goal classification from classify_goal().

        Returns:
            StrengthSnapshot with all current values.
        """
        return StrengthSnapshot(
            a_H=self.a_H,
            a_A=self.a_A,
            a_H_init=self.a_H_init,
            a_A_init=self.a_A_init,
            n_H=self.n_H,
            n_A=self.n_A,
            shrink_H=self._shrink_factor(self.n_H * 1.0),  # use last mu proxy
            shrink_A=self._shrink_factor(self.n_A * 1.0),
            classification=classification,
        )

    def _bayesian_update(
        self,
        a_prior: float,
        n_actual: int,
        mu_elapsed: float,
    ) -> float:
        """Apply the shrinkage formula for one team.

        shrink = mu_elapsed / (mu_elapsed + sigma_a^2)
        a_new  = a_prior + shrink * log((n_actual + 0.5) / (mu_elapsed + 0.5))

        Args:
            a_prior: Initial log-intensity (a_H_init or a_A_init).
            n_actual: Goals scored by this team so far.
            mu_elapsed: Expected goals elapsed for this team.

        Returns:
            Updated log-intensity.
        """
        if mu_elapsed <= 0.0:
            return a_prior

        shrink = mu_elapsed / (mu_elapsed + self.sigma_a_sq)
        correction = math.log((n_actual + 0.5) / (mu_elapsed + 0.5))
        correction = max(-0.3, min(0.3, correction))

        return a_prior + shrink * correction

    def _shrink_factor(self, mu_elapsed: float) -> float:
        """Compute the shrinkage factor for a given mu_elapsed."""
        if mu_elapsed <= 0.0:
            return 0.0
        return mu_elapsed / (mu_elapsed + self.sigma_a_sq)
