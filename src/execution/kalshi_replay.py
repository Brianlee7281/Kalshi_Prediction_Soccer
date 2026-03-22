"""Replays recorded Kalshi orderbook data to provide real P_kalshi.

Maintains a running orderbook state from snapshots + deltas recorded in
``data/recordings/{match_id}/kalshi_ob.jsonl``.  Provides mid-price and
execution price (with real spread from book) at any wall-clock time.

Mid-price logic matches ``src/engine/kalshi_ob_sync.py``:
    best_bid  = highest price in YES side with quantity > 0
    best_ask  = 1.0 - highest price in NO side with quantity > 0
    mid       = (best_bid + best_ask) / 2
"""

from __future__ import annotations

import json
from pathlib import Path


class KalshiOrderbookReplay:
    """Replays recorded Kalshi orderbook data for backtesting.

    Uses Option B from the prompt: maintains running orderbook state and
    advances forward as ticks progress (ticks are monotonic in time).
    Keeps a cursor into the message list so we never re-parse.
    """

    def __init__(
        self,
        kalshi_jsonl_path: Path,
        ticker_to_market: dict[str, str],
    ) -> None:
        self._path = kalshi_jsonl_path
        self._ticker_to_market = ticker_to_market  # ticker_str → market_type

        # Running orderbook: {ticker: {"yes": {price_str: qty_float}, "no": {price_str: qty_float}}}
        self._books: dict[str, dict[str, dict[str, float]]] = {}
        # Cached mid prices: {market_type: float}
        self._mids: dict[str, float] = {}

        # Pre-loaded messages sorted by _ts_wall
        self._messages: list[tuple[float, dict]] = []
        self._cursor: int = 0

        self._load()

    # ── loading ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Parse kalshi.jsonl once, store (ts_wall, msg) pairs."""
        msgs: list[tuple[float, dict]] = []
        with open(self._path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_type = obj.get("type", "")
                if msg_type not in ("orderbook_snapshot", "orderbook_delta"):
                    continue
                ts = obj.get("_ts_wall")
                if ts is None:
                    continue
                msgs.append((float(ts), obj))

        # Already chronological from recording, but sort defensively
        msgs.sort(key=lambda x: x[0])
        self._messages = msgs

    # ── public API ────────────────────────────────────────────────

    def get_prices_at(self, wall_clock_time: float) -> dict[str, float]:
        """Get mid prices for all markets at a specific wall-clock time.

        Advances the internal cursor, applying all messages up to
        ``wall_clock_time``.  Returns {market_type: mid_price}.
        """
        self._advance_to(wall_clock_time)
        return dict(self._mids)

    def get_spread(self, market_type: str) -> float:
        """Current bid-ask spread for a market."""
        for ticker, mt in self._ticker_to_market.items():
            if mt != market_type:
                continue
            book = self._books.get(ticker)
            if book is None:
                return 0.04  # fallback
            bid = self._best_bid(book)
            ask = self._best_ask(book)
            if bid is not None and ask is not None:
                return max(0.0, ask - bid)
        return 0.04

    def get_execution_price(
        self, market_type: str, direction: str, quantity: int = 1
    ) -> float:
        """Get realistic execution price from actual book depth.

        Walks the orderbook to fill ``quantity`` contracts.
        BUY_YES: walk the NO side (ascending NO price = descending YES ask).
        BUY_NO:  walk the YES side (descending YES price = ascending NO ask).

        Returns VWAP for the fill.  Falls back to mid if book is empty.
        """
        ticker = None
        for tk, mt in self._ticker_to_market.items():
            if mt == market_type:
                ticker = tk
                break
        if ticker is None or ticker not in self._books:
            return self._mids.get(market_type, 0.50)

        book = self._books[ticker]

        if direction == "BUY_YES":
            # Walk NO side: each NO bid at price p is a YES ask at 1-p.
            # Sort NO bids descending by price → ascending YES ask.
            no_levels = sorted(book["no"].items(), key=lambda x: -float(x[0]))
            total_qty = 0.0
            total_cost = 0.0
            for price_str, qty in no_levels:
                if qty <= 0:
                    continue
                yes_ask = 1.0 - float(price_str)
                take = min(qty, quantity - total_qty)
                total_cost += take * yes_ask
                total_qty += take
                if total_qty >= quantity:
                    break
            if total_qty > 0:
                return total_cost / total_qty
        else:  # BUY_NO
            # Walk YES side: each YES bid at price p is a NO ask at 1-p.
            # Sort YES bids descending by price → ascending NO ask.
            yes_levels = sorted(book["yes"].items(), key=lambda x: -float(x[0]))
            total_qty = 0.0
            total_cost = 0.0
            for price_str, qty in yes_levels:
                if qty <= 0:
                    continue
                no_ask = 1.0 - float(price_str)
                take = min(qty, quantity - total_qty)
                total_cost += take * no_ask
                total_qty += take
                if total_qty >= quantity:
                    break
            if total_qty > 0:
                return total_cost / total_qty

        return self._mids.get(market_type, 0.50)

    @property
    def available_markets(self) -> list[str]:
        """Market types that have at least one orderbook snapshot."""
        return list(self._mids.keys())

    # ── internals ─────────────────────────────────────────────────

    def _advance_to(self, wall_clock_time: float) -> None:
        """Apply all messages from cursor up to wall_clock_time."""
        while self._cursor < len(self._messages):
            ts, obj = self._messages[self._cursor]
            if ts > wall_clock_time:
                break
            self._apply(obj)
            self._cursor += 1

    def _apply(self, obj: dict) -> None:
        """Apply a single orderbook message to the running state."""
        msg_type = obj["type"]
        msg = obj["msg"]
        ticker = msg["market_ticker"]

        if msg_type == "orderbook_snapshot":
            self._apply_snapshot(ticker, msg)
        elif msg_type == "orderbook_delta":
            self._apply_delta(ticker, msg)

        # Recompute mid for this ticker
        self._recompute_mid(ticker)

    def _apply_snapshot(self, ticker: str, msg: dict) -> None:
        """Replace entire orderbook for a ticker from snapshot."""
        yes_book: dict[str, float] = {}
        for price_str, qty_str in msg.get("yes_dollars_fp", []):
            qty = float(qty_str)
            if qty > 0:
                yes_book[price_str] = qty

        no_book: dict[str, float] = {}
        for price_str, qty_str in msg.get("no_dollars_fp", []):
            qty = float(qty_str)
            if qty > 0:
                no_book[price_str] = qty

        self._books[ticker] = {"yes": yes_book, "no": no_book}

    def _apply_delta(self, ticker: str, msg: dict) -> None:
        """Apply incremental delta to the running orderbook."""
        if ticker not in self._books:
            return  # No snapshot yet; skip delta

        side = msg.get("side", "")
        if side not in ("yes", "no"):
            return

        price_str = msg.get("price_dollars", "")
        delta = float(msg.get("delta_fp", "0"))
        if not price_str:
            return

        book_side = self._books[ticker][side]
        current = book_side.get(price_str, 0.0)
        new_qty = current + delta
        if new_qty > 0:
            book_side[price_str] = new_qty
        else:
            book_side.pop(price_str, None)

    def _recompute_mid(self, ticker: str) -> None:
        """Recompute mid-price for a ticker, matching kalshi_ob_sync.py logic."""
        market_type = self._ticker_to_market.get(ticker)
        if market_type is None:
            return

        book = self._books.get(ticker)
        if book is None:
            return

        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            mid = best_bid
        elif best_ask is not None:
            mid = best_ask
        else:
            return

        self._mids[market_type] = mid

    @staticmethod
    def _best_bid(book: dict[str, dict[str, float]]) -> float | None:
        """Highest YES price with quantity > 0."""
        yes = book.get("yes", {})
        if not yes:
            return None
        return max(float(p) for p, q in yes.items() if q > 0)

    @staticmethod
    def _best_ask(book: dict[str, dict[str, float]]) -> float | None:
        """1.0 - highest NO price with quantity > 0."""
        no = book.get("no", {})
        if not no:
            return None
        best_no = max(float(p) for p, q in no.items() if q > 0)
        return 1.0 - best_no
