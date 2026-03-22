"""Backfill p_kalshi into an existing ticks.jsonl using latency recording orderbook data.

Joins recorded ticks (with correct P_model but no p_kalshi) to recorded orderbook
data (from a separate latency recording of the same match) via match time.

Usage:
  PYTHONPATH=. python scripts/backfill_p_kalshi.py \
    --ticks data/recordings/KXEPLGAME-26MAR20BOUMUN/ticks.jsonl \
    --latency-dir data/latency/KXEPLGAME-26MAR20BOUMUN
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

from scripts.run_phase3 import _tickers_list_to_dict
from src.execution.kalshi_replay import KalshiOrderbookReplay


def _load_time_mapping(kalshi_live_path: Path) -> list[tuple[float, float]]:
    """Build (match_t, wall_clock) mapping from kalshi_live.jsonl.

    Only includes records where status == "live" (match in progress).
    Returns sorted by match_t.
    """
    # Keep only the FIRST wall_clock per integer match_t.
    # Without dedup, bisect picks the LAST duplicate → interpolation spans
    # ~1s instead of ~60s → all ticks in a minute get the same OB state.
    first_seen: dict[int, float] = {}
    with open(kalshi_live_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("status") != "live":
                continue
            ts_wall = record.get("_ts_wall")
            minute = record.get("minute", 0)
            stoppage = record.get("stoppage", 0)
            if ts_wall is None:
                continue
            match_t_int = minute + stoppage
            if match_t_int not in first_seen:
                first_seen[match_t_int] = float(ts_wall)

    return [(float(k), v) for k, v in sorted(first_seen.items())]


def _match_t_to_wall_clock(
    match_t: float, mapping: list[tuple[float, float]]
) -> float | None:
    """Convert match time to wall-clock time via binary search + interpolation."""
    if not mapping:
        return None

    match_times = [m[0] for m in mapping]
    idx = bisect.bisect_right(match_times, match_t) - 1

    if idx < 0:
        return mapping[0][1]
    if idx >= len(mapping) - 1:
        return mapping[-1][1]

    # Linear interpolation between two nearest points
    t0, w0 = mapping[idx]
    t1, w1 = mapping[idx + 1]
    if t1 == t0:
        return w0
    frac = (match_t - t0) / (t1 - t0)
    return w0 + frac * (w1 - w0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill p_kalshi into ticks.jsonl")
    parser.add_argument("--ticks", required=True, help="Path to ticks.jsonl")
    parser.add_argument("--latency-dir", required=True,
                        help="Path to latency recording dir (with kalshi.jsonl and kalshi_live.jsonl)")
    parser.add_argument("--output", help="Output path (default: overwrite ticks.jsonl in-place)")
    args = parser.parse_args()

    ticks_path = Path(args.ticks)
    latency_dir = Path(args.latency_dir)
    output_path = Path(args.output) if args.output else ticks_path

    # Load metadata for ticker mapping
    meta_path = latency_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found in {latency_dir}", file=sys.stderr)
        sys.exit(1)
    with open(meta_path) as f:
        metadata = json.load(f)

    event_ticker = metadata.get("event_ticker", "")
    raw_tickers = metadata.get("kalshi_tickers", [])
    if not raw_tickers:
        print("ERROR: no kalshi_tickers in metadata", file=sys.stderr)
        sys.exit(1)

    # Build ticker → market_type mapping (KalshiOrderbookReplay expects this)
    market_to_ticker = _tickers_list_to_dict(event_ticker, raw_tickers)
    ticker_to_market = {v: k for k, v in market_to_ticker.items()}
    print(f"Ticker mapping: {ticker_to_market}")

    # Step 1: Build match-time → wall-clock mapping
    kalshi_live_path = latency_dir / "kalshi_live.jsonl"
    if not kalshi_live_path.exists():
        print(f"ERROR: kalshi_live.jsonl not found in {latency_dir}", file=sys.stderr)
        sys.exit(1)

    time_mapping = _load_time_mapping(kalshi_live_path)
    print(f"Time mapping: {len(time_mapping)} points, "
          f"match_t range [{time_mapping[0][0]:.1f}, {time_mapping[-1][0]:.1f}]")

    # Step 2: Load orderbook replay
    kalshi_ob_path = latency_dir / "kalshi.jsonl"
    if not kalshi_ob_path.exists():
        print(f"ERROR: kalshi.jsonl not found in {latency_dir}", file=sys.stderr)
        sys.exit(1)

    ob_replay = KalshiOrderbookReplay(kalshi_ob_path, ticker_to_market)
    print(f"Orderbook replay loaded, markets: {ob_replay.available_markets}")

    # Step 3: Process ticks
    with open(ticks_path, encoding="utf-8") as f:
        ticks = [json.loads(line) for line in f]

    enriched = 0
    empty = 0
    for tick in ticks:
        match_t = tick.get("t", 0.0)
        wall_clock = _match_t_to_wall_clock(match_t, time_mapping)
        if wall_clock is None:
            empty += 1
            continue

        p_kalshi = ob_replay.get_prices_at(wall_clock)
        if p_kalshi:
            tick["p_kalshi"] = p_kalshi
            enriched += 1
        else:
            empty += 1

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for tick in ticks:
            f.write(json.dumps(tick, default=str) + "\n")

    print(f"Done: {enriched} ticks enriched, {empty} without p_kalshi, "
          f"written to {output_path}")


if __name__ == "__main__":
    main()
