"""Plot P_model vs P_kalshi with trade markers for any replay result.

Usage:
    # Regular replay (all files in one directory):
    python scripts/plot_replay.py data/replay_results/KXEPLGAME-26MAR20BOUMUN

    # Tick replay (source data in separate directories):
    python scripts/plot_replay.py data/tick_replay_results/KXEPLGAME-26MAR20BOUMUN \
        --ticks data/recordings/KXEPLGAME-26MAR20BOUMUN \
        --orderbook data/latency/KXEPLGAME-26MAR20BOUMUN
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Data loading ─────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_clock_offset(tick_dir: Path) -> float:
    """Compute offset between recordings _ts and latency _ts_wall.

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


def _reconstruct_ob_from_raw(
    raw_path: Path,
    clock_offset: float,
    ticks: list[dict],
) -> list[dict]:
    """Reconstruct {_ts, ticker, mid} records from raw latency kalshi.jsonl.

    Uses _LocalBook to maintain orderbook state from snapshots + deltas.
    Maps wall clock to tick time using the clock offset:
        tick._ts = wall_clock - clock_offset
    Then maps tick._ts to match time t using the ticks themselves.

    Returns records with _ts = tick._ts (same clock as recordings ticks).
    """
    from src.engine.kalshi_ob_sync import _LocalBook

    # Build _ts → t lookup from ticks for plotting alignment
    # (the plot aligns OB _ts against tick _ts when ob_ts_is_match_time=False)
    # So we just need to output _ts in recordings clock = wall - offset.

    books: dict[str, _LocalBook] = {}
    result: list[dict] = []
    sample_count = 0

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ts_wall = r.get("_ts_wall", 0.0)
            rtype = r.get("type", "")
            msg = r.get("msg", {})
            ticker = msg.get("market_ticker", "")
            if not ticker:
                continue

            if ticker not in books:
                books[ticker] = _LocalBook()
            book = books[ticker]

            if rtype == "orderbook_snapshot":
                book.apply_snapshot_fp(
                    msg.get("yes_dollars_fp", []),
                    msg.get("no_dollars_fp", []),
                )
            elif rtype == "orderbook_delta":
                side = msg.get("side")
                price = msg.get("price_dollars")
                delta = msg.get("delta_fp")
                if side and price is not None and delta is not None:
                    book.apply_delta(side, float(price), float(delta))
            else:
                continue

            # Sample every 50th record to keep size manageable for plotting
            sample_count += 1
            if sample_count % 50 != 0:
                continue

            mid = book.mid()
            if mid is not None:
                # Convert wall clock to recordings _ts clock
                rec_ts = ts_wall - clock_offset
                result.append({"_ts": rec_ts, "ticker": ticker, "mid": mid})

    return result


# ── Ticker → market type mapping ────────────────────────────────

def _map_tickers(
    ob_records: list[dict],
    ticks: list[dict],
    trades: list[dict],
) -> dict[str, str]:
    """Derive {ticker: market_type} from data, not hardcoded.

    *-TIE → draw always.
    For the other two, use trades' entry_reason or correlate with P_model at t=0.
    """
    # Collect unique tickers
    tickers = sorted({r.get("ticker", "") for r in ob_records if r.get("ticker")})
    if not tickers:
        return {}

    mapping: dict[str, str] = {}

    # TIE is always draw
    for t in tickers:
        suffix = t.split("-")[-1].upper()
        if suffix == "TIE":
            mapping[t] = "draw"

    # Try trades' entry_reason first
    reason_map: dict[str, str] = {}
    for trade in trades:
        ticker = trade.get("ticker", "")
        reason = trade.get("entry_reason", "")
        if ticker and reason and ticker not in mapping:
            reason_map[ticker] = reason

    for ticker, reason in reason_map.items():
        if reason in ("home_win", "draw", "away_win"):
            mapping[ticker] = reason

    # If still unmapped, correlate with P_model at t=0
    unmapped = [t for t in tickers if t not in mapping]
    if unmapped and ticks:
        first_pm = ticks[0].get("P_model", {})
        # Collect first mid for each unmapped ticker
        first_mids: dict[str, float] = {}
        for r in ob_records:
            ticker = r.get("ticker", "")
            if ticker in unmapped and ticker not in first_mids:
                first_mids[ticker] = r.get("mid", 0.5)
            if len(first_mids) == len(unmapped):
                break

        remaining_types = [mt for mt in ("home_win", "away_win") if mt not in mapping.values()]
        for ticker in unmapped:
            mid = first_mids.get(ticker, 0.5)
            best_type = None
            best_diff = float("inf")
            for mt in remaining_types:
                pm = first_pm.get(mt, 0.5)
                diff = abs(mid - pm)
                if diff < best_diff:
                    best_diff = diff
                    best_type = mt
            if best_type:
                mapping[ticker] = best_type
                remaining_types.remove(best_type)

    return mapping


# ── Build P_kalshi time series aligned to tick times ─────────────

def _build_kalshi_series(
    ob_records: list[dict],
    ticks: list[dict],
    ticker_to_market: dict[str, str],
    ob_ts_is_match_time: bool = False,
) -> dict[str, tuple[list[float], list[float]]]:
    """For each market, build (times[], mids[]) aligned to tick times.

    If ob_ts_is_match_time=False (default), ob _ts is replay-seconds and we
    align to tick _ts. If True (reconstructed from raw), ob _ts is already
    match minutes and we align to tick t directly.
    """
    tick_ts_list = [tk.get("_ts", 0.0) for tk in ticks]
    tick_t_list = [tk.get("t", 0.0) for tk in ticks]

    # Group ob records by market, sorted by _ts
    market_obs: dict[str, list[tuple[float, float]]] = {}
    for r in ob_records:
        ticker = r.get("ticker", "")
        mt = ticker_to_market.get(ticker)
        if mt is None:
            continue
        ts = r.get("_ts", 0.0)
        mid = r.get("mid", 0.0)
        if mt not in market_obs:
            market_obs[mt] = []
        market_obs[mt].append((ts, mid))

    for mt in market_obs:
        market_obs[mt].sort(key=lambda x: x[0])

    # For each market, sample at tick times
    result: dict[str, tuple[list[float], list[float]]] = {}
    for mt, obs in market_obs.items():
        ob_ts = [o[0] for o in obs]
        ob_mid = [o[1] for o in obs]
        times: list[float] = []
        mids: list[float] = []

        # Choose the alignment clock
        align_keys = tick_t_list if ob_ts_is_match_time else tick_ts_list

        for align_key, tick_t in zip(align_keys, tick_t_list):
            idx = bisect_right(ob_ts, align_key) - 1
            if idx >= 0:
                times.append(tick_t)
                mids.append(ob_mid[idx])
        result[mt] = (times, mids)

    return result


# ── Main plot ────────────────────────────────────────────────────

def plot_replay(
    replay_dir: Path,
    ticks_dir: Path | None = None,
    orderbook_dir: Path | None = None,
) -> None:
    # Load trades + metadata from replay_dir (always)
    trades = _load_jsonl(replay_dir / "trades.jsonl")
    metadata = _load_metadata(replay_dir / "metadata.json")

    # Load ticks: from --ticks dir if provided, else replay_dir
    tick_source = ticks_dir if ticks_dir else replay_dir
    ticks = _load_jsonl(tick_source / "ticks.jsonl")

    # Load events: try tick source first (recordings has events), then replay_dir
    events = _load_jsonl(tick_source / "events.jsonl")
    if not events:
        events = _load_jsonl(replay_dir / "events.jsonl")

    # Load orderbook: from --orderbook dir (raw, needs reconstruction) or replay_dir
    if orderbook_dir and (orderbook_dir / "kalshi.jsonl").exists():
        clock_offset = _compute_clock_offset(tick_source)
        ob_records = _reconstruct_ob_from_raw(
            orderbook_dir / "kalshi.jsonl",
            clock_offset,
            ticks,
        )
        # Reconstructed _ts is in recordings clock (same as ticks._ts)
        ob_ts_is_match_time = False
    else:
        ob_records = _load_jsonl(replay_dir / "kalshi_ob.jsonl")
        ob_ts_is_match_time = False

    # Merge metadata from tick source if replay_dir metadata is sparse
    if ticks_dir:
        tick_meta = _load_metadata(ticks_dir / "metadata.json")
        for k, v in tick_meta.items():
            if k not in metadata:
                metadata[k] = v
    if orderbook_dir:
        ob_meta = _load_metadata(orderbook_dir / "metadata.json")
        for k, v in ob_meta.items():
            if k not in metadata:
                metadata[k] = v

    if not ticks:
        print("No ticks found.")
        return

    # Ticker mapping
    ticker_to_market = _map_tickers(ob_records, ticks, trades)
    market_to_ticker: dict[str, str] = {v: k for k, v in ticker_to_market.items()}

    # Extract team codes from tickers
    team_codes: dict[str, str] = {}
    for ticker, mt in ticker_to_market.items():
        team_codes[mt] = ticker.split("-")[-1]

    # P_model series
    markets = ["home_win", "draw", "away_win"]
    tick_times = [tk.get("t", 0.0) for tk in ticks]
    p_model: dict[str, list[float]] = {}
    sigma_mc: dict[str, list[float]] = {}
    for mt in markets:
        p_model[mt] = [tk.get("P_model", {}).get(mt, 0.0) for tk in ticks]
        sigma_mc[mt] = [tk.get("sigma_MC", {}).get(mt, 0.0) for tk in ticks]

    # P_kalshi series (sampled at tick times)
    kalshi_series = _build_kalshi_series(
        ob_records, ticks, ticker_to_market, ob_ts_is_match_time,
    )

    # Goals and period changes
    goals = [e for e in events if e.get("type") == "goal"]
    halftime_events = [e for e in events if e.get("type") == "period_change"
                       and e.get("new_phase") == "SECOND_HALF"]

    # Map trades to markets
    trade_market: dict[int, str] = {}
    for trade in trades:
        ticker = trade.get("ticker", "")
        mt = ticker_to_market.get(ticker)
        if mt:
            trade_market[trade.get("id", 0)] = mt

    # Final score from last tick or last goal
    last_score = ticks[-1].get("score", [0, 0]) if ticks else [0, 0]

    # Trade summary
    total_pnl = sum(t.get("realized_pnl", 0.0) or 0.0 for t in trades)

    # ── Build figure ──────────────────────────────────────────
    match_id = metadata.get("match_id", metadata.get("event_ticker", replay_dir.name))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, dpi=150)
    fig.suptitle(match_id, fontsize=14, fontweight="bold")

    for ax_idx, mt in enumerate(markets):
        ax = axes[ax_idx]
        code = team_codes.get(mt, mt)
        title = {"home_win": f"Home Win ({code})", "draw": "Draw",
                 "away_win": f"Away Win ({code})"}[mt]
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.3)

        # P_model line + sigma band
        pm = p_model[mt]
        sm = sigma_mc[mt]
        pm_upper = [p + s for p, s in zip(pm, sm)]
        pm_lower = [p - s for p, s in zip(pm, sm)]
        ax.plot(tick_times, pm, color="#2563eb", linewidth=1.2, label="P_model", zorder=3)
        ax.fill_between(tick_times, pm_lower, pm_upper, color="#2563eb", alpha=0.12,
                        label="\u00b11\u03c3 MC", zorder=2)

        # P_kalshi line
        if mt in kalshi_series:
            kt, km = kalshi_series[mt]
            ax.plot(kt, km, color="#dc2626", linewidth=1.0, label="P_kalshi", zorder=3)

            # Edge shading
            if len(kt) == len(tick_times):
                ax.fill_between(
                    tick_times, pm, km,
                    where=[p > k for p, k in zip(pm, km)],
                    color="green", alpha=0.08, interpolate=True,
                )
                ax.fill_between(
                    tick_times, pm, km,
                    where=[p < k for p, k in zip(pm, km)],
                    color="red", alpha=0.08, interpolate=True,
                )

        # Goal lines (on all subplots)
        for goal in goals:
            gt = goal.get("t", goal.get("model_t", 0.0))
            sa = goal.get("score_after", [0, 0])
            minute = goal.get("minute", int(gt))
            ax.axvline(gt, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
            ax.text(gt + 0.3, 0.95, f"{sa[0]}-{sa[1]} ({minute}')",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7, color="gray", va="top")

        # Halftime line
        for ht in halftime_events:
            ht_t = ht.get("model_t", 45.0)
            ax.axvline(ht_t, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
            ax.text(ht_t + 0.3, 0.02, "HT", transform=ax.get_xaxis_transform(),
                    fontsize=7, color="gray")

        # Trade markers
        for trade in trades:
            tid = trade.get("id", 0)
            if trade_market.get(tid) != mt:
                continue

            entry_tick = trade.get("entry_tick", 0)
            exit_tick = trade.get("exit_tick", 0)
            entry_price = trade.get("entry_price", 0.0)
            exit_price = trade.get("exit_price")
            direction = trade.get("direction", "BUY_YES")
            pnl = trade.get("realized_pnl")
            qty = trade.get("quantity", 0)

            # Map tick index to match time
            entry_t = ticks[entry_tick - 1]["t"] if 0 < entry_tick <= len(ticks) else 0
            exit_t = ticks[exit_tick - 1]["t"] if exit_tick and 0 < exit_tick <= len(ticks) else None

            # Entry marker
            marker = "^" if direction == "BUY_YES" else "v"
            color = "#16a34a" if direction == "BUY_YES" else "#dc2626"
            ax.scatter(entry_t, entry_price, marker=marker, color=color,
                       s=80, zorder=5, edgecolors="black", linewidths=0.5)

            # Exit marker + connecting line
            if exit_price is not None and exit_t is not None:
                ax.scatter(exit_t, exit_price, marker="X", color="#dc2626",
                           s=60, zorder=5, edgecolors="black", linewidths=0.5)
                ax.plot([entry_t, exit_t], [entry_price, exit_price],
                        color="gray", linestyle="--", linewidth=0.7, alpha=0.6, zorder=4)

                # PnL annotation
                if pnl is not None:
                    pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
                    ax.annotate(
                        f"{qty}c ${pnl:+.2f}",
                        xy=(exit_t, exit_price),
                        xytext=(5, 8), textcoords="offset points",
                        fontsize=6, color=pnl_color,
                        fontweight="bold",
                    )

        # Legend on first subplot only
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    # X-axis label
    axes[-1].set_xlabel("Match Minute")
    max_t = max(tick_times) if tick_times else 90
    axes[-1].set_xlim(0, max_t + 2)

    # ── Summary text box ──────────────────────────────────────
    summary_lines = [
        f"Final: {last_score[0]}-{last_score[1]}  |  "
        f"{len(trades)} trades  |  Total PnL: ${total_pnl:+.2f}"
    ]
    for trade in trades:
        market_code = trade.get("ticker", "?").split("-")[-1]
        d = trade.get("direction", "?")
        q = trade.get("quantity", 0)
        ep = trade.get("entry_price", 0.0)
        xp = trade.get("exit_price")
        reason = trade.get("exit_reason", "?")
        pnl = trade.get("realized_pnl")
        xp_str = f"{xp:.4f}" if xp is not None else "open"
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "-"
        summary_lines.append(
            f"  {market_code} {d} {q}c @ {ep:.4f} -> {xp_str} ({reason}) {pnl_str}"
        )

    summary_text = "\n".join(summary_lines)
    fig.text(
        0.02, -0.01, summary_text,
        fontsize=7, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.02 + 0.012 * len(trades), 1, 0.96])

    # Save
    out_path = replay_dir / "trade_visualization.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Match: {match_id}")
    print(f"Trades: {len(trades)}  |  Total PnL: ${total_pnl:+.2f}")
    print(f"Saved: {out_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot replay trade visualization")
    parser.add_argument("replay_dir", type=str, help="Path to replay result directory")
    parser.add_argument("--ticks", type=str, default=None,
                        help="Path to recordings dir with ticks.jsonl (for tick replay)")
    parser.add_argument("--orderbook", type=str, default=None,
                        help="Path to latency dir with kalshi.jsonl (for tick replay)")
    args = parser.parse_args()
    plot_replay(
        Path(args.replay_dir),
        ticks_dir=Path(args.ticks) if args.ticks else None,
        orderbook_dir=Path(args.orderbook) if args.orderbook else None,
    )


if __name__ == "__main__":
    main()
