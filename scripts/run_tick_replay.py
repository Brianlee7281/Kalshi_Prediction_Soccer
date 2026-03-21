"""Replay recorded live ticks through Phase 4 execution with live orderbook.

Reads pre-recorded Phase 3 ticks (1-second resolution) and pairs them with
the recorded Kalshi orderbook to run Phase 4 paper trading at full fidelity.

No Phase 3 re-computation — uses the exact P_model values from the live run.

Usage:
  PYTHONPATH=. python scripts/run_tick_replay.py \
      --ticks data/recordings/KXEPLGAME-26MAR20BOUMUN \
      --orderbook data/latency/KXEPLGAME-26MAR20BOUMUN
  PYTHONPATH=. python scripts/run_tick_replay.py \
      --ticks data/recordings/KXEPLGAME-26MAR20BOUMUN \
      --orderbook data/latency/KXEPLGAME-26MAR20BOUMUN \
      --bankroll 5000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

from src.common.types import MarketProbs, MatchPnL, TickPayload, TradingMode
from src.engine.kalshi_ob_sync import _LocalBook
from src.execution.execution_loop import execution_loop
from src.execution.mock_db import MockDBPool

log = structlog.get_logger("run_tick_replay")


# ── Data loading ─────────────────────────────────────────────────

def _load_ticks(tick_dir: Path) -> list[dict]:
    """Load recorded ticks from a recordings directory."""
    path = tick_dir / "ticks.jsonl"
    if not path.exists():
        log.error("ticks_not_found", path=str(path))
        sys.exit(1)
    ticks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ticks.append(json.loads(line))
    ticks.sort(key=lambda r: r.get("_ts", 0.0))
    return ticks


def _load_orderbook(ob_dir: Path) -> list[tuple[float, str, dict]]:
    """Load orderbook records as (ts_wall, type, msg) tuples."""
    path = ob_dir / "kalshi.jsonl"
    if not path.exists():
        log.error("orderbook_not_found", path=str(path))
        sys.exit(1)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ts = r.get("_ts_wall", 0.0)
            rtype = r.get("type", "")
            msg = r.get("msg", {})
            records.append((ts, rtype, msg))
    records.sort(key=lambda x: x[0])
    return records


def _compute_clock_offset(tick_dir: Path) -> float:
    """Compute the offset between recordings _ts and latency _ts_wall.

    Uses goal events which have both _ts (recordings monotonic) and
    occurence_ts (Kalshi wall clock / unix epoch).
    Returns offset such that: wall_clock = recordings._ts + offset.
    """
    events_path = tick_dir / "events.jsonl"
    if not events_path.exists():
        return 0.0
    offsets: list[float] = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("type") == "goal" and e.get("occurence_ts") and e.get("_ts"):
                offsets.append(e["occurence_ts"] - e["_ts"])
    if not offsets:
        return 0.0
    return sum(offsets) / len(offsets)


def _load_metadata(d: Path) -> dict:
    path = d / "metadata.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _tickers_list_to_dict(event_ticker: str, tickers: list[str]) -> dict[str, str]:
    """Convert flat ticker list to {market_type: ticker} dict."""
    parts = event_ticker.split("-")
    match_abbr = parts[-1] if parts else ""
    home_abbr = match_abbr[-6:-3].upper()
    away_abbr = match_abbr[-3:].upper()

    result: dict[str, str] = {}
    for t in tickers:
        suffix = t.split("-")[-1].upper()
        if suffix == "TIE":
            result["draw"] = t
        elif suffix == home_abbr:
            result["home_win"] = t
        elif suffix == away_abbr:
            result["away_win"] = t
    return result


# ── Orderbook sync ───────────────────────────────────────────────

class OrderbookPlayer:
    """Replays orderbook records up to a given wall-clock timestamp."""

    def __init__(
        self,
        records: list[tuple[float, str, dict]],
        ticker_to_market: dict[str, str],
    ) -> None:
        self._records = records
        self._index = 0
        self._ticker_to_market = ticker_to_market
        self._books: dict[str, _LocalBook] = {}

    def advance_to(self, wall_ts: float) -> dict[str, float]:
        """Apply all orderbook records up to wall_ts, return current p_kalshi."""
        while self._index < len(self._records):
            ts, rtype, msg = self._records[self._index]
            if ts > wall_ts:
                break
            self._index += 1

            ticker = msg.get("market_ticker", "")
            if ticker not in self._ticker_to_market:
                continue

            if ticker not in self._books:
                self._books[ticker] = _LocalBook()
            book = self._books[ticker]

            if rtype == "orderbook_snapshot":
                yes_fp = msg.get("yes_dollars_fp", [])
                no_fp = msg.get("no_dollars_fp", [])
                if yes_fp or no_fp:
                    book.apply_snapshot_fp(yes_fp, no_fp)
            elif rtype == "orderbook_delta":
                side = msg.get("side")
                price = msg.get("price_dollars")
                delta = msg.get("delta_fp")
                if side and price is not None and delta is not None:
                    book.apply_delta(side, float(price), float(delta))

        # Build p_kalshi from current book state
        p_kalshi: dict[str, float] = {}
        for ticker, market_type in self._ticker_to_market.items():
            if ticker in self._books:
                mid = self._books[ticker].mid()
                if mid is not None:
                    p_kalshi[market_type] = mid
        return p_kalshi


# ── Tick → TickPayload conversion ────────────────────────────────

def _tick_to_payload(tick: dict) -> TickPayload:
    """Convert a recorded tick dict to a TickPayload."""
    pm = tick.get("P_model", {})
    sm = tick.get("sigma_MC", {})
    score = tick.get("score", [0, 0])

    return TickPayload(
        match_id=tick.get("match_id", ""),
        t=tick.get("t", 0.0),
        engine_phase=tick.get("engine_phase", "FIRST_HALF"),
        P_model=MarketProbs(
            home_win=pm.get("home_win", 0.0),
            draw=pm.get("draw", 0.0),
            away_win=pm.get("away_win", 0.0),
            over_25=pm.get("over_25", 0.0),
            btts_yes=pm.get("btts_yes", 0.0),
        ),
        sigma_MC=MarketProbs(
            home_win=sm.get("home_win", 0.0),
            draw=sm.get("draw", 0.0),
            away_win=sm.get("away_win", 0.0),
            over_25=sm.get("over_25", 0.0),
            btts_yes=sm.get("btts_yes", 0.0),
        ),
        score=tuple(score),
        X=tick.get("X", 0),
        delta_S=tick.get("delta_S", 0),
        mu_H=tick.get("mu_H", 0.0),
        mu_A=tick.get("mu_A", 0.0),
        a_H_current=tick.get("a_H_current", 0.0),
        a_A_current=tick.get("a_A_current", 0.0),
        last_goal_type=tick.get("last_goal_type", "NEUTRAL"),
        ekf_P_H=tick.get("ekf_P_H", 0.0),
        ekf_P_A=tick.get("ekf_P_A", 0.0),
        hmm_state=tick.get("hmm_state", 0),
        dom_index=tick.get("dom_index", 0.0),
        surprise_score=tick.get("surprise_score", 0.0),
        # Override cooldown/order_allowed from recorded data — the live run
        # had a tick-based cooldown that blocked post-goal trading.  Our current
        # model removes that cooldown (post-goal is where edges appear).
        # Only respect ob_freeze (VAR/penalty — genuinely unreliable orderbook).
        order_allowed=not tick.get("ob_freeze", False),
        cooldown=False,
        ob_freeze=tick.get("ob_freeze", False),
        event_state="IDLE",
    )


# ── Main ─────────────────────────────────────────────────────────

async def run_tick_replay(
    tick_dir: Path,
    ob_dir: Path,
    bankroll: float = 10_000.0,
) -> None:
    """Replay recorded ticks + orderbook through Phase 4."""
    # Load data
    ticks = _load_ticks(tick_dir)
    ob_records = _load_orderbook(ob_dir)
    clock_offset = _compute_clock_offset(tick_dir)

    # Metadata
    ob_meta = _load_metadata(ob_dir)
    tick_meta = _load_metadata(tick_dir)
    event_ticker = ob_meta.get(
        "event_ticker", tick_meta.get("match_id", tick_dir.name)
    )
    raw_tickers = ob_meta.get("kalshi_tickers", [])
    kalshi_tickers = _tickers_list_to_dict(event_ticker, raw_tickers)
    ticker_to_market = {v: k for k, v in kalshi_tickers.items()}
    home_team = ob_meta.get("home_team", "Home")
    away_team = ob_meta.get("away_team", "Away")

    log.info(
        "tick_replay_start",
        ticks=len(ticks),
        ob_records=len(ob_records),
        kalshi_tickers=kalshi_tickers,
        bankroll=bankroll,
    )

    # Orderbook player
    ob_player = OrderbookPlayer(ob_records, ticker_to_market)

    # Mock model object for execution_loop (it reads p_kalshi and kalshi_tickers)
    class _MockModel:
        def __init__(self) -> None:
            self.match_id = event_ticker
            self.p_kalshi: dict[str, float] = {}
            self.kalshi_tickers = kalshi_tickers

    model = _MockModel()

    # Phase 4 setup
    db_pool = MockDBPool(initial_bankroll=bankroll)
    phase4_queue: asyncio.Queue = asyncio.Queue()

    exec_task = asyncio.create_task(
        execution_loop(phase4_queue, model, db_pool, TradingMode.PAPER)
    )

    # Feed ticks
    for tick in ticks:
        tick_ts = tick.get("_ts", 0.0)
        wall_ts = tick_ts + clock_offset

        # Advance orderbook to this tick's time
        model.p_kalshi = ob_player.advance_to(wall_ts)

        # Convert and enqueue
        payload = _tick_to_payload(tick)
        await phase4_queue.put(payload)

        # Yield to let execution_loop process
        await asyncio.sleep(0)

    # Send FINISHED sentinel
    _zero = MarketProbs(home_win=0.0, draw=0.0, away_win=0.0)
    last_tick = ticks[-1] if ticks else {}
    last_score = tuple(last_tick.get("score", [0, 0]))
    finished = TickPayload(
        match_id=event_ticker, t=last_tick.get("t", 0.0),
        engine_phase="FINISHED",
        P_model=_zero, sigma_MC=_zero,
        score=last_score, X=0, delta_S=0,
        mu_H=0.0, mu_A=0.0, a_H_current=0.0, a_A_current=0.0,
        ekf_P_H=0.0, ekf_P_A=0.0, hmm_state=0, dom_index=0.0,
        surprise_score=0.0, order_allowed=False, cooldown=False,
        ob_freeze=False, event_state="none",
    )
    await phase4_queue.put(finished)
    match_pnl: MatchPnL = await exec_task

    # Output
    output_dir = Path("data/tick_replay_results")
    result_dir = output_dir / event_ticker
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(result_dir / "pnl_summary.json", "w", encoding="utf-8") as f:
        json.dump(match_pnl.model_dump(), f, indent=2, default=str)
    with open(result_dir / "trades.jsonl", "w", encoding="utf-8") as f:
        for pos in db_pool.positions:
            f.write(json.dumps(pos, default=str) + "\n")
    with open(result_dir / "bankroll.jsonl", "w", encoding="utf-8") as f:
        for snap in db_pool.snapshots:
            f.write(json.dumps(snap, default=str) + "\n")

    # Print summary
    lines: list[str] = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  Phase 4 Tick Replay — {home_team} vs {away_team}")
    lines.append(f"  {len(ticks)} ticks @ ~1s resolution")
    lines.append(f"{'='*65}")

    lines.append(f"\n  Bankroll: {bankroll:,.2f} -> {db_pool.bankroll:,.2f}")

    closed = [p for p in db_pool.positions if p.get("status") == "CLOSED"]
    total_realized = sum(p.get("realized_pnl", 0.0) or 0.0 for p in closed)
    wins = sum(1 for p in closed if (p.get("realized_pnl") or 0) > 0)
    losses = sum(1 for p in closed if (p.get("realized_pnl") or 0) < 0)

    lines.append(f"  Trades: {len(closed)}  |  Wins: {wins}  |  Losses: {losses}")
    lines.append(f"  Realized PnL: {total_realized:+,.2f}")
    if closed:
        roi = (total_realized / bankroll) * 100
        lines.append(f"  ROI: {roi:+.2f}%")

    if db_pool.positions:
        lines.append(f"\n  {'Market':<12} {'Dir':<8} {'Qty':>4} {'Entry':>7} {'Exit':>7} {'PnL':>8} {'Reason'}")
        lines.append(f"  {'-'*65}")
        for pos in db_pool.positions:
            market = pos.get("ticker", "?").split("-")[-1]
            direction = pos.get("direction", "?")
            qty = pos.get("quantity", 0)
            entry = pos.get("entry_price", 0.0)
            exit_p = pos.get("exit_price")
            pnl = pos.get("realized_pnl")
            reason = pos.get("exit_reason", "?")
            exit_str = f"{exit_p:.4f}" if exit_p is not None else "  open"
            pnl_str = f"{pnl:+.2f}" if pnl is not None else "    -"
            lines.append(f"  {market:<12} {direction:<8} {qty:>4} {entry:.4f} {exit_str} {pnl_str} {reason}")

    lines.append("")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

    log.info(
        "tick_replay_done",
        ticks=len(ticks),
        trades=len(closed),
        realized_pnl=round(total_realized, 2),
        final_score=last_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded ticks through Phase 4")
    parser.add_argument("--ticks", type=str, required=True,
                        help="Path to recordings directory (with ticks.jsonl)")
    parser.add_argument("--orderbook", type=str, required=True,
                        help="Path to latency directory (with kalshi.jsonl)")
    parser.add_argument("--bankroll", type=float, default=10_000.0,
                        help="Starting bankroll (default: $10,000)")
    args = parser.parse_args()

    asyncio.run(run_tick_replay(
        Path(args.ticks), Path(args.orderbook), args.bankroll,
    ))


if __name__ == "__main__":
    main()
