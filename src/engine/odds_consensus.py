"""OddsConsensus — aggregates live bookmaker odds into a single reference price.

Combines up to 5 bookmaker feeds with Betfair Exchange receiving 2x weight
in a weighted median calculation. Detects staleness (>10s since last update)
and sudden coordinated moves that may indicate an in-match event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.common.logging import get_logger
from src.common.types import BookmakerState, MarketProbs, OddsConsensusResult

logger = get_logger("engine.odds_consensus")

# Market fields to compute consensus over
_MARKET_FIELDS = ("home_win", "draw", "away_win", "over_25", "under_25", "btts_yes", "btts_no")

# Event detection constants
_EVENT_MOVE_THRESHOLD = 0.03  # 3% move
_EVENT_WINDOW_S = 5.0  # within 5 seconds


@dataclass
class _BookmakerEntry:
    """Internal mutable state for a single bookmaker source."""

    name: str
    implied: MarketProbs
    prev_implied: MarketProbs | None = None
    last_update: float = 0.0  # time.monotonic()
    prev_update: float = 0.0  # time.monotonic() of previous update


class OddsConsensus:
    """Aggregates live odds from 5 bookmakers into consensus reference price."""

    BOOKMAKERS = ["Betfair Exchange", "Bet365", "1xbet", "Sbobet", "DraftKings"]
    WEIGHTS: dict[str, float] = {"Betfair Exchange": 2.0}  # default weight = 1.0
    STALE_THRESHOLD_S = 10.0

    def __init__(self) -> None:
        self.sources: dict[str, _BookmakerEntry] = {}

    def update_bookmaker(self, name: str, implied: MarketProbs) -> None:
        """Update a bookmaker's odds. Called by odds_api_listener on each WS message."""
        now = time.monotonic()
        existing = self.sources.get(name)

        if existing is not None:
            existing.prev_implied = existing.implied
            existing.prev_update = existing.last_update
            existing.implied = implied
            existing.last_update = now
        else:
            self.sources[name] = _BookmakerEntry(
                name=name,
                implied=implied,
                last_update=now,
            )

        logger.debug(
            "bookmaker_updated",
            bookmaker=name,
            home_win=implied.home_win,
            draw=implied.draw,
            away_win=implied.away_win,
        )

    def compute_reference(self) -> OddsConsensusResult:
        """Compute consensus from all fresh (non-stale) sources.

        - fresh = sources where last_update < STALE_THRESHOLD_S ago
        - 0 fresh: confidence = NONE
        - 1 fresh: confidence = LOW
        - 2+ fresh: confidence = HIGH
        - P_consensus = weighted median (Betfair 2x weight)
        - event_detected = 2+ sources moved >3% same direction within 5s
        """
        now = time.monotonic()

        fresh: list[_BookmakerEntry] = []
        bookmaker_states: list[BookmakerState] = []

        for entry in self.sources.values():
            is_stale = (now - entry.last_update) > self.STALE_THRESHOLD_S
            bookmaker_states.append(
                BookmakerState(
                    name=entry.name,
                    implied=entry.implied,
                    last_update=datetime.fromtimestamp(
                        time.time() - (now - entry.last_update), tz=timezone.utc
                    ),
                    is_stale=is_stale,
                )
            )
            if not is_stale:
                fresh.append(entry)

        n_fresh = len(fresh)

        if n_fresh == 0:
            confidence = "NONE"
        elif n_fresh == 1:
            confidence = "LOW"
        else:
            confidence = "HIGH"

        # Compute per-market weighted median from fresh sources
        consensus_dict: dict[str, float | None] = {}
        for mkt in _MARKET_FIELDS:
            values: list[float] = []
            weights: list[float] = []
            for entry in fresh:
                val = getattr(entry.implied, mkt)
                if val is not None:
                    values.append(val)
                    weights.append(self.WEIGHTS.get(entry.name, 1.0))

            if values:
                consensus_dict[mkt] = self._weighted_median(values, weights)
            else:
                consensus_dict[mkt] = None

        # Ensure required fields have defaults
        P_consensus = MarketProbs(
            home_win=consensus_dict.get("home_win") or 0.0,
            draw=consensus_dict.get("draw") or 0.0,
            away_win=consensus_dict.get("away_win") or 0.0,
            over_25=consensus_dict.get("over_25"),
            under_25=consensus_dict.get("under_25"),
            btts_yes=consensus_dict.get("btts_yes"),
            btts_no=consensus_dict.get("btts_no"),
        )

        event_detected = self._detect_event(fresh, now)

        return OddsConsensusResult(
            P_consensus=P_consensus,
            confidence=confidence,
            n_fresh_sources=n_fresh,
            bookmakers=bookmaker_states,
            event_detected=event_detected,
        )

    @staticmethod
    def _weighted_median(values: list[float], weights: list[float]) -> float:
        """Weighted median: expand each value by its weight, take median.

        For integer-like weights this is equivalent to repeating each value
        weight times. For fractional weights we sort by value and find the
        point where cumulative weight crosses 50%.
        """
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]

        # Sort by value
        pairs = sorted(zip(values, weights))
        total_weight = sum(w for _, w in pairs)
        half = total_weight / 2.0

        cumulative = 0.0
        for i, (val, w) in enumerate(pairs):
            cumulative += w
            if cumulative >= half:
                # If cumulative lands exactly on half and there's a next value,
                # average the two (standard median behavior)
                if cumulative == half and i + 1 < len(pairs):
                    return (val + pairs[i + 1][0]) / 2.0
                return val

        return pairs[-1][0]  # fallback

    @staticmethod
    def _detect_event(
        fresh: list[_BookmakerEntry], now: float
    ) -> bool:
        """Check if 2+ bookmakers moved >3% in same direction within 5s.

        Compares current vs previous implied for home_win on each fresh source
        that was updated within the event window.
        """
        up_count = 0
        down_count = 0

        for entry in fresh:
            if entry.prev_implied is None:
                continue
            # Only consider recent updates
            if (now - entry.last_update) > _EVENT_WINDOW_S:
                continue

            diff = entry.implied.home_win - entry.prev_implied.home_win
            if diff > _EVENT_MOVE_THRESHOLD:
                up_count += 1
            elif diff < -_EVENT_MOVE_THRESHOLD:
                down_count += 1

        return up_count >= 2 or down_count >= 2
