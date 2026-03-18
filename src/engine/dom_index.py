"""DomIndex — goal-based dominance index with exponential decay.

Layer 2 fallback when HMM is not available. Computes a momentum
signal from recent goals: home goals add +1, away goals add -1,
each decaying exponentially with half-life ~7 minutes (κ=0.1/min).

DomIndex(t) = Σ_home exp(-κ(t-t_g)) - Σ_away exp(-κ(t-t_g))
momentum_state = tanh(DomIndex) ∈ (-1, +1)
"""

from __future__ import annotations

import math


class DomIndex:
    """Goal-based dominance index with exponential decay."""

    DECAY_RATE: float = 0.1  # κ per minute, half-life ≈ 6.9 min

    def __init__(self) -> None:
        self._goals: list[tuple[float, str]] = []  # (time_minutes, "home"|"away")

    def record_goal(self, t: float, team: str) -> None:
        """Record a goal at time t (minutes)."""
        self._goals.append((t, team))

    def compute(self, t: float) -> float:
        """Compute raw DomIndex value at time t."""
        value = 0.0
        for t_g, team in self._goals:
            weight = math.exp(-self.DECAY_RATE * (t - t_g))
            if team == "home":
                value += weight
            else:
                value -= weight
        return value

    def momentum_state(self, t: float) -> float:
        """Returns tanh(DomIndex) ∈ (-1, +1)."""
        return math.tanh(self.compute(t))

    def quantized_state(self, t: float) -> int:
        """Returns -1, 0, or +1 based on momentum."""
        m = self.momentum_state(t)
        if m > 0.3:
            return 1
        elif m < -0.3:
            return -1
        return 0
