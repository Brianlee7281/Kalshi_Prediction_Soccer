"""Simulates Kalshi market prices as an independent market estimate.

Key design principles:
1. P_kalshi has its OWN random walk — not derived from P_model + noise
2. Both P_kalshi and P_model converge to the same true probability over time
3. After events, P_kalshi reacts gradually (exponential catch-up, not freeze+jump)
4. Execution involves spread + slippage that scales with post-event volatility

In live trading, kalshi_ob_sync.py maintains model.p_kalshi from the
real orderbook WebSocket. In backtesting, we replace it with this
simulator.
"""

from __future__ import annotations

import math

import numpy as np

from src.common.types import MarketProbs

MARKET_TYPES: list[str] = ["home_win", "draw", "away_win", "over_25", "btts_yes"]


class KalshiPriceSimulator:
    """Simulates Kalshi prices as an independent market estimate.

    After events: P_kalshi reacts gradually — an instant partial reaction
    (market_initial_reaction) followed by exponential catch-up with
    half-life market_reaction_half_life ticks.

    Between events: P_kalshi follows an independent random walk with
    very weak drift toward P_model. No mean-reversion — trading against
    noise is 50/50, not free money.

    Spread widens after events and decays back to base_spread.
    """

    def __init__(
        self,
        # Event reaction parameters
        market_reaction_half_life: float = 5.0,
        market_initial_reaction: float = 0.3,
        # Between-event noise
        tick_volatility: float = 0.002,
        # Execution friction
        base_spread: float = 0.02,
        event_spread_mult: float = 3.0,
        event_spread_decay: float = 0.1,
        # Slow information flow between events
        between_event_drift: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.market_reaction_half_life = market_reaction_half_life
        self.market_initial_reaction = market_initial_reaction
        self.tick_volatility = tick_volatility
        self.base_spread = base_spread
        self.event_spread_mult = event_spread_mult
        self.event_spread_decay = event_spread_decay
        self.between_event_drift = between_event_drift
        self.rng = np.random.default_rng(seed)

        # Per-tick catch-up fraction derived from half-life:
        # After half_life ticks, 50% of gap is closed.
        # (1 - alpha)^half_life = 0.5  =>  alpha = 1 - 0.5^(1/half_life)
        self._catch_up_alpha = 1.0 - 0.5 ** (1.0 / max(market_reaction_half_life, 0.1))

        # Internal state per market
        self.p_kalshi: dict[str, float] = {}
        # The "target" that P_kalshi is catching up toward after an event.
        # Between events this is just the last P_model seen on the event tick.
        self._event_target: dict[str, float] = {}
        self.last_event_tick: int = -1000
        self.current_tick: int = 0

    # ── public API ────────────────────────────────────────────────

    def initialize(self, kickoff_probs: MarketProbs) -> dict[str, float]:
        """Set initial Kalshi prices at kickoff.

        At kickoff, Kalshi prices approximately equal market odds.
        Small independent noise so there's no instant edge at t=0.
        """
        for mt in MARKET_TYPES:
            p = getattr(kickoff_probs, mt)
            if p is None:
                continue
            noise = self.rng.normal(0.0, self.tick_volatility * 2)
            self.p_kalshi[mt] = float(np.clip(p + noise, 0.01, 0.99))
            self._event_target[mt] = p

        return dict(self.p_kalshi)

    def update(
        self,
        tick: int,
        p_model: MarketProbs,
        is_event_tick: bool = False,
    ) -> dict[str, float]:
        """Update simulated Kalshi prices for this tick.

        On event tick: P_kalshi jumps by market_initial_reaction fraction
        of the gap, then continues exponential catch-up.

        After event: P_kalshi closes the gap exponentially with half-life
        market_reaction_half_life ticks, plus independent noise.

        Between events: independent random walk with very weak drift
        toward P_model (no profitable mean-reversion).
        """
        self.current_tick = tick

        if is_event_tick:
            self.last_event_tick = tick
            # Instant partial reaction: market captures some of the move
            for mt in MARKET_TYPES:
                p_m = getattr(p_model, mt)
                if p_m is None or mt not in self.p_kalshi:
                    continue
                pre_event = self.p_kalshi[mt]
                jump = (p_m - pre_event) * self.market_initial_reaction
                self.p_kalshi[mt] = float(np.clip(pre_event + jump, 0.01, 0.99))
                self._event_target[mt] = p_m
            return dict(self.p_kalshi)

        ticks_since_event = tick - self.last_event_tick

        for mt in MARKET_TYPES:
            p_m = getattr(p_model, mt)
            if p_m is None or mt not in self.p_kalshi:
                continue

            # The event target evolves with P_model (model may keep updating
            # a_H/a_A via EKF after the goal, so the "true" target moves).
            self._event_target[mt] = p_m

            if ticks_since_event <= self.market_reaction_half_life * 4:
                # Post-event catch-up: exponential convergence toward
                # current P_model with independent noise.
                gap = p_m - self.p_kalshi[mt]
                catch_up = gap * self._catch_up_alpha
                noise = self.rng.normal(0.0, self.tick_volatility)
                new_p = self.p_kalshi[mt] + catch_up + noise
            else:
                # Between events: independent random walk.
                # Very weak drift toward P_model represents slow
                # information flow — NOT mean-reversion to exploit.
                innovation = self.rng.normal(0.0, self.tick_volatility)
                drift = (p_m - self.p_kalshi[mt]) * self.between_event_drift
                new_p = self.p_kalshi[mt] + innovation + drift

            self.p_kalshi[mt] = float(np.clip(new_p, 0.01, 0.99))

        return dict(self.p_kalshi)

    def get_execution_price(self, market_type: str, direction: str) -> float:
        """Execution price including spread + event volatility premium.

        Spread widens after events (market makers increase spread when
        uncertain), then decays exponentially back to base_spread.
        """
        mid = self.p_kalshi.get(market_type, 0.5)
        current_spread = self._current_spread()
        half_spread = current_spread / 2
        if direction == "BUY_YES":
            return float(np.clip(mid + half_spread, 0.01, 0.99))
        else:
            return float(np.clip(mid - half_spread, 0.01, 0.99))

    def get_prices(self) -> dict[str, float]:
        """Return current mid prices for all markets."""
        return dict(self.p_kalshi)

    # ── internals ─────────────────────────────────────────────────

    def _current_spread(self) -> float:
        """Compute current spread accounting for event volatility."""
        ticks_since = self.current_tick - self.last_event_tick
        event_premium = (self.event_spread_mult - 1.0) * math.exp(
            -self.event_spread_decay * max(ticks_since, 0)
        )
        return self.base_spread * (1.0 + event_premium)
