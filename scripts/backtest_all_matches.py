"""Replay-test Phase 4 trading on ALL recorded matches and plot results.

Runs run_tick_replay on each match in data/recordings/, collects PnL
and trade data, then generates summary plots:
  1. Per-match PnL bar chart
  2. Cumulative PnL equity curve (matches played chronologically)
  3. Trade-level scatter (edge vs realized PnL)
  4. Per-match P_model vs P_kalshi subplots with trade markers

Usage:
    PYTHONPATH=. python scripts/backtest_all_matches.py
    PYTHONPATH=. python scripts/backtest_all_matches.py --bankroll 5000
    PYTHONPATH=. python scripts/backtest_all_matches.py --skip-replay  # plot from cached results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RECORDINGS_DIR = Path("data/recordings")
RESULTS_DIR = Path("data/tick_replay_results")


# ── Run replay on a single match ──────────────────────────────────

async def replay_one_match(
    rec_dir: Path,
    bankroll: float,
) -> dict:
    """Run tick replay on one match, return summary dict."""
    from scripts.run_tick_replay import run_tick_replay

    match_id = rec_dir.name
    result = {
        "match_id": match_id,
        "rec_dir": str(rec_dir),
        "success": False,
        "total_pnl": 0.0,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "bankroll_start": bankroll,
        "bankroll_end": bankroll,
        "trades": [],
        "home_team": "",
        "away_team": "",
        "final_score": [0, 0],
    }

    # Check required files
    if not (rec_dir / "ticks.jsonl").exists():
        print(f"  SKIP {match_id}: no ticks.jsonl")
        return result
    if not (rec_dir / "kalshi_ob.jsonl").exists():
        print(f"  SKIP {match_id}: no kalshi_ob.jsonl")
        return result

    try:
        await run_tick_replay(rec_dir, rec_dir, bankroll)
        result["success"] = True
    except Exception as exc:
        print(f"  FAIL {match_id}: {exc}")
        traceback.print_exc()
        return result

    # Read results from saved files
    result_dir = RESULTS_DIR / _find_result_dir_name(rec_dir)
    pnl_path = result_dir / "pnl_summary.json"
    trades_path = result_dir / "trades.jsonl"
    bankroll_path = result_dir / "bankroll.jsonl"

    if pnl_path.exists():
        with open(pnl_path, encoding="utf-8") as f:
            pnl_data = json.load(f)
        result["total_pnl"] = pnl_data.get("total_pnl", 0.0)
        result["trade_count"] = pnl_data.get("trade_count", 0)
        result["win_count"] = pnl_data.get("win_count", 0)
        result["loss_count"] = pnl_data.get("loss_count", 0)

    if trades_path.exists():
        trades = []
        with open(trades_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        result["trades"] = trades

        # Recompute from trades for consistency
        closed = [t for t in trades if t.get("status") == "CLOSED"]
        result["trade_count"] = len(closed)
        result["win_count"] = sum(1 for t in closed if (t.get("realized_pnl") or 0) > 0)
        result["loss_count"] = sum(1 for t in closed if (t.get("realized_pnl") or 0) < 0)
        result["total_pnl"] = sum(t.get("realized_pnl", 0.0) or 0.0 for t in closed)

    if bankroll_path.exists():
        snaps = []
        with open(bankroll_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    snaps.append(json.loads(line))
        if snaps:
            result["bankroll_end"] = snaps[-1].get("balance", bankroll)

    # Load metadata
    meta_path = rec_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        result["home_team"] = meta.get("home_team", "")
        result["away_team"] = meta.get("away_team", "")

    # Final score from events
    events_path = rec_dir / "events.jsonl"
    if events_path.exists():
        last_score = [0, 0]
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("type") == "goal":
                    last_score = e.get("score_after", last_score)
        result["final_score"] = last_score

    return result


def _find_result_dir_name(rec_dir: Path) -> str:
    """Find the result directory name (matches event_ticker from metadata)."""
    meta_path = rec_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("event_ticker", meta.get("match_id", rec_dir.name))
    return rec_dir.name


# ── Load cached results (--skip-replay) ──────────────────────────

def load_cached_results(bankroll: float) -> list[dict]:
    """Load results from previous replay runs."""
    results = []
    for rec_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not rec_dir.is_dir():
            continue
        match_id = rec_dir.name
        result_name = _find_result_dir_name(rec_dir)
        result_dir = RESULTS_DIR / result_name

        result = {
            "match_id": match_id,
            "rec_dir": str(rec_dir),
            "success": False,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "bankroll_start": bankroll,
            "bankroll_end": bankroll,
            "trades": [],
            "home_team": "",
            "away_team": "",
            "final_score": [0, 0],
        }

        trades_path = result_dir / "trades.jsonl"
        if not trades_path.exists():
            continue

        result["success"] = True
        trades = []
        with open(trades_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        result["trades"] = trades
        closed = [t for t in trades if t.get("status") == "CLOSED"]
        result["trade_count"] = len(closed)
        result["win_count"] = sum(1 for t in closed if (t.get("realized_pnl") or 0) > 0)
        result["loss_count"] = sum(1 for t in closed if (t.get("realized_pnl") or 0) < 0)
        result["total_pnl"] = sum(t.get("realized_pnl", 0.0) or 0.0 for t in closed)

        meta_path = rec_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            result["home_team"] = meta.get("home_team", "")
            result["away_team"] = meta.get("away_team", "")

        events_path = rec_dir / "events.jsonl"
        if events_path.exists():
            last_score = [0, 0]
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e.get("type") == "goal":
                        last_score = e.get("score_after", last_score)
            result["final_score"] = last_score

        results.append(result)
    return results


# ── Plotting ──────────────────────────────────────────────────────

def _short_label(r: dict) -> str:
    """Short match label for x-axis."""
    h = r.get("home_team", "")[:8]
    a = r.get("away_team", "")[:8]
    if h and a:
        return f"{h} v {a}"
    return r["match_id"].split("-")[-1]


def plot_summary(results: list[dict], bankroll: float) -> None:
    """Generate 4-panel summary of all matches."""
    # Filter to successful replays with trades
    valid = [r for r in results if r["success"]]
    if not valid:
        print("No valid results to plot.")
        return

    labels = [_short_label(r) for r in valid]
    pnls = [r["total_pnl"] for r in valid]
    trade_counts = [r["trade_count"] for r in valid]
    win_counts = [r["win_count"] for r in valid]
    loss_counts = [r["loss_count"] for r in valid]
    cumulative_pnl = list(np.cumsum(pnls))

    # Collect all individual trades
    all_trades = []
    for r in valid:
        for t in r.get("trades", []):
            if t.get("status") == "CLOSED" and t.get("realized_pnl") is not None:
                all_trades.append(t)

    fig = plt.figure(figsize=(20, 14), dpi=150)
    fig.suptitle(
        f"Phase 4 Backtest — {len(valid)} Matches | "
        f"Bankroll: ${bankroll:,.0f}",
        fontsize=16, fontweight="bold",
    )

    # Panel 1: Per-match PnL bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]
    bars = ax1.bar(range(len(valid)), pnls, color=colors, edgecolor="black", linewidth=0.3)
    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("PnL ($)")
    ax1.set_title("Per-Match PnL", fontsize=12)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.3)
    # Annotate bars
    for i, (bar, pnl) in enumerate(zip(bars, pnls)):
        if pnl != 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, pnl,
                f"${pnl:+.0f}", ha="center",
                va="bottom" if pnl > 0 else "top",
                fontsize=6, fontweight="bold",
            )

    # Panel 2: Cumulative PnL equity curve
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(range(len(valid)), cumulative_pnl, "b-o", linewidth=2, markersize=4)
    ax2.fill_between(
        range(len(valid)), cumulative_pnl, 0,
        where=[p >= 0 for p in cumulative_pnl], color="green", alpha=0.1,
    )
    ax2.fill_between(
        range(len(valid)), cumulative_pnl, 0,
        where=[p < 0 for p in cumulative_pnl], color="red", alpha=0.1,
    )
    ax2.set_xticks(range(len(valid)))
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("Cumulative PnL ($)")
    ax2.set_title("Equity Curve", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(alpha=0.3)

    # Panel 3: Trade-level PnL histogram
    ax3 = fig.add_subplot(2, 2, 3)
    if all_trades:
        trade_pnls = [t["realized_pnl"] for t in all_trades]
        trade_edges = [
            abs(t.get("entry_price", 0.5) - 0.5)
            for t in all_trades
        ]
        colors_trades = ["#16a34a" if p >= 0 else "#dc2626" for p in trade_pnls]
        ax3.hist(trade_pnls, bins=30, color="#2563eb", edgecolor="black", linewidth=0.3, alpha=0.7)
        ax3.axvline(0, color="black", linewidth=1)
        mean_pnl = np.mean(trade_pnls)
        median_pnl = np.median(trade_pnls)
        ax3.axvline(mean_pnl, color="red", linestyle="--", linewidth=1, label=f"Mean: ${mean_pnl:+.2f}")
        ax3.axvline(median_pnl, color="orange", linestyle="--", linewidth=1, label=f"Median: ${median_pnl:+.2f}")
        ax3.legend(fontsize=8)
    ax3.set_xlabel("Trade PnL ($)")
    ax3.set_ylabel("Count")
    ax3.set_title("Trade PnL Distribution", fontsize=12)
    ax3.grid(alpha=0.3)

    # Panel 4: Win rate + trade count by match
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(len(valid))
    width = 0.35
    ax4.bar(x - width / 2, win_counts, width, color="#16a34a", label="Wins", edgecolor="black", linewidth=0.3)
    ax4.bar(x + width / 2, loss_counts, width, color="#dc2626", label="Losses", edgecolor="black", linewidth=0.3)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax4.set_ylabel("Trade Count")
    ax4.set_title("Wins vs Losses per Match", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    # Summary stats text
    total_pnl = sum(pnls)
    total_trades = sum(trade_counts)
    total_wins = sum(win_counts)
    total_losses = sum(loss_counts)
    win_rate = total_wins / total_trades * 100 if total_trades else 0
    roi = total_pnl / bankroll * 100
    profitable_matches = sum(1 for p in pnls if p > 0)
    losing_matches = sum(1 for p in pnls if p < 0)
    no_trade_matches = sum(1 for tc in trade_counts if tc == 0)

    summary = (
        f"Total PnL: ${total_pnl:+,.2f}  |  ROI: {roi:+.2f}%  |  "
        f"Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%  |  "
        f"Profitable Matches: {profitable_matches}/{len(valid)}  |  "
        f"No-Trade Matches: {no_trade_matches}"
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    out_path = RESULTS_DIR / "backtest_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSummary plot saved: {out_path}")


def plot_individual_matches(results: list[dict]) -> None:
    """Generate per-match P_model vs P_kalshi with trade markers."""
    valid = [r for r in results if r["success"]]
    if not valid:
        return

    # Determine grid size
    n = len(valid)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, 4.5 * rows), dpi=120)
    fig.suptitle("Per-Match: P_model (blue) vs P_kalshi (red) — Home Win Market",
                 fontsize=14, fontweight="bold")

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, r in enumerate(valid):
        ax = axes[idx]
        rec_dir = Path(r["rec_dir"])
        ticks_path = rec_dir / "ticks.jsonl"

        if not ticks_path.exists():
            ax.set_title(_short_label(r), fontsize=9)
            ax.text(0.5, 0.5, "No ticks", ha="center", va="center", transform=ax.transAxes)
            continue

        # Load ticks (lightweight — only extract what we need)
        times = []
        pm_hw = []
        pk_hw = []
        with open(ticks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tk = json.loads(line)
                times.append(tk.get("t", 0.0))
                pm_hw.append(tk.get("P_model", {}).get("home_win", 0.0))
                pk = tk.get("p_kalshi", {})
                pk_hw.append(pk.get("home_win") if pk else None)

        # Plot P_model
        ax.plot(times, pm_hw, color="#2563eb", linewidth=0.8, label="P_model")

        # Plot P_kalshi if available
        pk_times = [t for t, p in zip(times, pk_hw) if p is not None]
        pk_vals = [p for p in pk_hw if p is not None]
        if pk_times:
            ax.plot(pk_times, pk_vals, color="#dc2626", linewidth=0.6, alpha=0.7, label="P_kalshi")

        # Goal markers from events
        events_path = rec_dir / "events.jsonl"
        if events_path.exists():
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e.get("type") == "goal":
                        gt = e.get("minute", 0)
                        sa = e.get("score_after", [0, 0])
                        ax.axvline(gt, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
                        ax.text(gt + 0.5, 0.97, f"{sa[0]}-{sa[1]}",
                                transform=ax.get_xaxis_transform(),
                                fontsize=5, color="gray", va="top")

        # Trade markers
        trades = r.get("trades", [])
        for trade in trades:
            if trade.get("entry_reason") != "home_win":
                continue
            entry_tick = trade.get("entry_tick", 0)
            entry_price = trade.get("entry_price", 0.0)
            pnl = trade.get("realized_pnl")
            if 0 < entry_tick <= len(times):
                entry_t = times[entry_tick - 1]
                marker = "^" if trade.get("direction") == "BUY_YES" else "v"
                color = "#16a34a" if (pnl or 0) >= 0 else "#dc2626"
                ax.scatter(entry_t, entry_price, marker=marker, color=color,
                           s=30, zorder=5, edgecolors="black", linewidths=0.3)

        score = r.get("final_score", [0, 0])
        pnl = r["total_pnl"]
        label = _short_label(r)
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
        ax.set_title(
            f"{label} ({score[0]}-{score[1]})  ${pnl:+.1f}",
            fontsize=8, color=pnl_color,
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 95)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused axes
    for idx in range(len(valid), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = RESULTS_DIR / "backtest_per_match.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Per-match plot saved: {out_path}")


def print_summary_table(results: list[dict], bankroll: float) -> None:
    """Print a text summary table of all matches."""
    valid = [r for r in results if r["success"]]
    if not valid:
        print("No valid results.")
        return

    print(f"\n{'='*90}")
    print(f"  PHASE 4 BACKTEST SUMMARY — {len(valid)} matches @ ${bankroll:,.0f} bankroll")
    print(f"{'='*90}")
    print(f"  {'Match':<30} {'Score':>6} {'Trades':>7} {'W':>3} {'L':>3} {'PnL':>10} {'ROI':>7}")
    print(f"  {'-'*90}")

    for r in valid:
        label = _short_label(r)
        score = r.get("final_score", [0, 0])
        tc = r["trade_count"]
        w = r["win_count"]
        lo = r["loss_count"]
        pnl = r["total_pnl"]
        roi = pnl / bankroll * 100
        print(f"  {label:<30} {score[0]}-{score[1]:>3} {tc:>7} {w:>3} {lo:>3} ${pnl:>+9.2f} {roi:>+6.2f}%")

    total_pnl = sum(r["total_pnl"] for r in valid)
    total_trades = sum(r["trade_count"] for r in valid)
    total_wins = sum(r["win_count"] for r in valid)
    total_losses = sum(r["loss_count"] for r in valid)
    total_roi = total_pnl / bankroll * 100
    win_rate = total_wins / total_trades * 100 if total_trades else 0

    print(f"  {'-'*90}")
    print(f"  {'TOTAL':<30} {'':>6} {total_trades:>7} {total_wins:>3} {total_losses:>3} ${total_pnl:>+9.2f} {total_roi:>+6.2f}%")
    print(f"\n  Win Rate: {win_rate:.1f}%  |  "
          f"Avg PnL/match: ${total_pnl/len(valid):+.2f}  |  "
          f"Avg trades/match: {total_trades/len(valid):.1f}")
    print(f"{'='*90}\n")


# ── Main ──────────────────────────────────────────────────────────

async def run_all(bankroll: float) -> list[dict]:
    """Run tick replay on all recorded matches sequentially."""
    rec_dirs = sorted(
        d for d in RECORDINGS_DIR.iterdir()
        if d.is_dir()
    )
    print(f"Found {len(rec_dirs)} recorded matches in {RECORDINGS_DIR}\n")

    results: list[dict] = []
    for i, rec_dir in enumerate(rec_dirs, 1):
        print(f"[{i}/{len(rec_dirs)}] {rec_dir.name}")
        result = await replay_one_match(rec_dir, bankroll)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay-test trading on all recorded matches"
    )
    parser.add_argument("--bankroll", type=float, default=10_000.0,
                        help="Starting bankroll per match (default: $10,000)")
    parser.add_argument("--skip-replay", action="store_true",
                        help="Skip replay, plot from cached results")
    args = parser.parse_args()

    if args.skip_replay:
        print("Loading cached results...")
        results = load_cached_results(args.bankroll)
    else:
        results = asyncio.run(run_all(args.bankroll))

    print_summary_table(results, args.bankroll)
    plot_summary(results, args.bankroll)
    plot_individual_matches(results)


if __name__ == "__main__":
    main()
