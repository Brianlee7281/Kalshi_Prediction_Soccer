"""HMM Estimator — Layer 2 tactical momentum (stub).

Gracefully degrades to DomIndex when HMM parameters are unavailable.
Full HMM implementation requires recorded match data (TODO).

Interface: update() with live_stats, record_goal(), state property,
adjust_intensity() for Layer 1 intensity modification.
"""

from __future__ import annotations

import math

from src.common.logging import get_logger
from src.engine.dom_index import DomIndex

logger = get_logger("engine.hmm_estimator")


class HMMEstimator:
    """Hidden Markov Model for tactical momentum.

    Degrades to DomIndex when hmm_params is None.
    """

    def __init__(self, hmm_params: dict | None = None) -> None:
        self._dom_index = DomIndex()
        self._hmm_available = hmm_params is not None
        self._warned = False
        self._current_t = 0.0

    def update(self, live_stats: dict | None, t: float) -> None:
        """Update momentum estimate from live statistics.

        Args:
            live_stats: Dict with shots_on_target_h/a, corners_h/a, possession_h.
                        None if stats unavailable this poll.
            t: Current match time in minutes.
        """
        self._current_t = t
        if not self._hmm_available:
            if not self._warned:
                logger.warning("hmm_not_trained", msg="Using DomIndex fallback")
                self._warned = True
            return
        # TODO: full HMM Baum-Welch forward pass with live_stats emissions

    def record_goal(self, t: float, team: str) -> None:
        """Record a goal for DomIndex tracking."""
        self._dom_index.record_goal(t, team)

    @property
    def state(self) -> int:
        """Returns -1 (away dominant), 0 (balanced), or +1 (home dominant)."""
        if self._hmm_available:
            return 0  # TODO: full HMM state
        return self._dom_index.quantized_state(self._current_t)

    @property
    def dom_index_value(self) -> float:
        """Returns raw DomIndex value (always available, even with HMM)."""
        return self._dom_index.compute(self._current_t)

    def adjust_intensity(
        self,
        lambda_H: float,
        lambda_A: float,
        phi_H: float = 0.0,
        phi_A: float = 0.0,
    ) -> tuple[float, float]:
        """Adjust intensities by momentum: λ_adj = λ × exp(φ × Z_t).

        When phi_H=phi_A=0 (default), returns unchanged intensities.
        """
        Z = self._dom_index.momentum_state(self._current_t)
        lambda_H_adj = lambda_H * math.exp(phi_H * Z)
        lambda_A_adj = lambda_A * math.exp(phi_A * Z)
        return lambda_H_adj, lambda_A_adj
