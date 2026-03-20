#!/usr/bin/env python3
"""Spike reversal & drawdown analysis for early goal detection.

Analyzes worst-case scenarios: orderbook spikes that fully reverse,
particularly VAR-disallowed goals and late-game false signals.

Quantifies:
1. All reversal events (spikes ≥8c that revert within 120s)
2. Per-contract drawdown at realistic entry/exit timing
3. Late-game amplification (75min+ vs 0-75min)
4. Dollar risk per match assuming Kelly-sized positions

Usage:
  PYTHONPATH=. python scripts/analyze_spike_reversals.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────

LATENCY_DIR = Path("data/latency")
OUTPUT_DIR = Path("data/analysis/spike_reversals")

SPIKE_THRESHOLD = 0.08  # 8 cents minimum spike to trigger detection
REVERT_WINDOW = 120     # seconds to check for full reversion
REVERT_FRACTION = 0.50  # price must retrace ≥50% of spike to count as reversal

# Entry timing: we'd detect the spike and enter at these offsets after spike start
ENTRY_DELAYS = [1, 2, 3, 5]  # seconds after spike begins

# Exit timing: how fast we can react to reversal + fill
EXIT_REACTION_TIME = 3  # seconds to detect reversal and submit exit order
EXIT_SLIPPAGE_CENTS = 2  # additional cents lost hitting the bid during a crash

# Kelly sizing assumptions
BANKROLL = 500.0       # dollars
KELLY_FRACTION = 0.10  # base Kelly
HALF_KELLY = 0.05      # provisional sizing
CONTRACT_PRICE_CENTS = 50  # typical mid-game contract price (50c)


def _utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _utc_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ─── Data loading (reuse from adjustment curve script) ──────────────


def load_metadata(match_dir: Path) -> dict:
    with open(match_dir / "metadata.json") as f:
        return json.load(f)


def load_events(match_dir: Path) -> list[dict]:
    events = []
    with open(match_dir / "events.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def identify_ticker_roles(meta: dict) -> dict[str, str]:
    tickers = meta.get("kalshi_tickers", [])
    roles = {}
    candidates = []
    for t in tickers:
        suffix = t.rsplit("-", 1)[-1]
        if suffix == "TIE":
            roles["draw"] = suffix
        elif suffix not in candidates:
            candidates.append(suffix)
    if not candidates or not tickers:
        return roles
    parts = tickers[0].split("-")
    if len(parts) >= 3:
        middle = parts[-2]
        team_codes = middle[-6:]
    else:
        team_codes = ""
    for c in candidates:
        if team_codes:
            idx = team_codes.find(c)
            if idx == 0:
                roles["home"] = c
            elif idx > 0:
                roles["away"] = c
            else:
                if "home" not in roles:
                    roles["home"] = c
                else:
                    roles["away"] = c
        else:
            if "home" not in roles:
                roles["home"] = c
            else:
                roles["away"] = c
    return roles


# ─── Orderbook reconstruction ──────────────────────────────────────


def build_all_orderbook_timelines(
    match_dir: Path, ticker_substrs: dict[str, str]
) -> dict[str, list[dict]]:
    """Rebuild best_yes price + spread for ALL tickers in a single file pass."""
    yes_books: dict[str, dict[str, float]] = {r: {} for r in ticker_substrs}
    no_books: dict[str, dict[str, float]] = {r: {} for r in ticker_substrs}
    timelines: dict[str, list[dict]] = {r: [] for r in ticker_substrs}
    substr_to_role: dict[str, str] = {v: k for k, v in ticker_substrs.items()}

    with open(match_dir / "kalshi.jsonl") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("msg", data)
            ticker = msg.get("market_ticker", "")
            if not ticker:
                continue
            role = None
            for substr, r in substr_to_role.items():
                if ticker.endswith(f"-{substr}"):
                    role = r
                    break
            if role is None:
                continue

            ts = data.get("_ts_wall", 0)
            msg_type = data.get("type", "")
            yes_book = yes_books[role]
            no_book = no_books[role]

            if msg_type == "orderbook_snapshot":
                yes_book.clear()
                no_book.clear()
                for entry in msg.get("yes_dollars_fp", []):
                    if len(entry) >= 2:
                        qty = float(entry[1])
                        if qty > 0:
                            yes_book[entry[0]] = qty
                for entry in msg.get("no_dollars_fp", []):
                    if len(entry) >= 2:
                        qty = float(entry[1])
                        if qty > 0:
                            no_book[entry[0]] = qty

            elif msg_type == "orderbook_delta":
                side = msg.get("side", "")
                price_str = msg.get("price_dollars", "")
                delta = float(msg.get("delta_fp", "0"))
                if not price_str:
                    continue
                book = yes_book if side == "yes" else no_book
                current = book.get(price_str, 0)
                new_qty = current + delta
                if new_qty > 0:
                    book[price_str] = new_qty
                else:
                    book.pop(price_str, None)
            else:
                continue

            best_yes = max((float(p) for p in yes_book), default=None) if yes_book else None
            best_no = max((float(p) for p in no_book), default=None) if no_book else None

            if best_yes is None:
                continue

            if best_no is not None:
                spread = max(0.0, (1.0 - best_no) - best_yes)
            else:
                spread = None

            # Also compute depth at best levels (contracts available)
            best_yes_str = max(yes_book.keys(), key=lambda p: float(p)) if yes_book else None
            depth_at_best = float(yes_book.get(best_yes_str, 0)) if best_yes_str else 0

            timelines[role].append({
                "ts_wall": ts,
                "mid": best_yes,
                "spread": spread,
                "best_yes": best_yes,
                "best_no": best_no,
                "depth": depth_at_best,
            })

    return timelines


def sample_price_at(timeline: list[dict], ts: float) -> dict | None:
    """Find most recent orderbook state at or before ts."""
    best = None
    for entry in timeline:
        if entry["ts_wall"] <= ts:
            best = entry
        elif entry["ts_wall"] > ts:
            break
    return best


def sample_prices_in_window(
    timeline: list[dict], ts_start: float, ts_end: float
) -> list[dict]:
    """Get all orderbook entries in [ts_start, ts_end]."""
    return [e for e in timeline if ts_start <= e["ts_wall"] <= ts_end]


# ─── Match minute estimation ────────────────────────────────────────


def estimate_match_minute(
    ts: float, events: list[dict], match_start_utc: str
) -> float | None:
    """Estimate the match minute from events timeline.

    Uses status_change events (which track the minute) to estimate
    the current match minute at any wall-clock timestamp.
    """
    # Build a list of (ts_wall, minute) from status changes
    minute_markers = []
    for evt in events:
        if evt["type"] == "status_change":
            status = evt.get("new_status", "")
            if status.isdigit():
                minute_markers.append((evt["ts_wall"], int(status)))

    if not minute_markers:
        return None

    # Find the closest markers before and after ts
    before = None
    after = None
    for marker_ts, minute in minute_markers:
        if marker_ts <= ts:
            before = (marker_ts, minute)
        elif after is None:
            after = (marker_ts, minute)
            break

    if before is not None:
        # Interpolate from last known minute
        elapsed = (ts - before[0]) / 60.0
        return before[1] + elapsed
    elif after is not None:
        elapsed = (after[0] - ts) / 60.0
        return after[1] - elapsed
    return None


# ─── Spike detection with reversal tracking ─────────────────────────


def find_all_spikes(
    timeline: list[dict],
    events: list[dict],
    goal_timestamps: list[float],
    match_id: str,
    market_role: str,
    match_start_utc: str,
) -> list[dict]:
    """Find ALL price spikes ≥ threshold, classify as goal or non-goal,
    and track whether they reversed.

    Returns detailed spike records including:
    - Peak price, timing, magnitude
    - Whether it reversed and how fast
    - Price trajectory at +1s, +3s, +5s, +10s, +30s, +60s, +120s
    - Estimated match minute
    - Whether this coincides with a known goal event
    """
    if len(timeline) < 10:
        return []

    spikes = []
    trail = 0
    i = 0

    while i < len(timeline):
        entry = timeline[i]
        ts = entry["ts_wall"]

        # Advance trailing baseline pointer (10s lookback for baseline)
        target_baseline = ts - 10.0
        while trail < i - 1 and timeline[trail + 1]["ts_wall"] <= target_baseline:
            trail += 1

        if timeline[trail]["ts_wall"] > target_baseline or trail >= i:
            i += 1
            continue

        baseline_mid = timeline[trail]["mid"]
        move = entry["mid"] - baseline_mid  # signed
        abs_move = abs(move)

        if abs_move < SPIKE_THRESHOLD:
            i += 1
            continue

        spike_start_ts = ts
        spike_direction = "up" if move > 0 else "down"

        # Track peak within the next 30s
        peak_price = entry["mid"]
        peak_ts = ts
        scan_entries = sample_prices_in_window(timeline, ts, ts + 30)
        for se in scan_entries:
            if spike_direction == "up" and se["mid"] > peak_price:
                peak_price = se["mid"]
                peak_ts = se["ts_wall"]
            elif spike_direction == "down" and se["mid"] < peak_price:
                peak_price = se["mid"]
                peak_ts = se["ts_wall"]

        peak_move = abs(peak_price - baseline_mid)

        # Sample price trajectory after spike
        trajectory = {}
        for offset in [1, 2, 3, 5, 10, 15, 30, 60, 90, 120]:
            sample = sample_price_at(timeline, ts + offset)
            if sample is not None:
                trajectory[offset] = {
                    "mid": sample["mid"],
                    "spread": sample["spread"],
                    "depth": sample.get("depth", 0),
                }

        # Check for reversion
        reverted = False
        revert_ts = None
        revert_speed = None
        min_revert_price = None

        # Scan forward for reversion (price returns to within 50% of baseline)
        revert_entries = sample_prices_in_window(timeline, ts, ts + REVERT_WINDOW)
        if spike_direction == "up":
            for re_entry in revert_entries:
                revert_amount = peak_price - re_entry["mid"]
                if revert_amount >= peak_move * REVERT_FRACTION:
                    reverted = True
                    revert_ts = re_entry["ts_wall"]
                    revert_speed = revert_ts - peak_ts
                    min_revert_price = re_entry["mid"]
                    break
        else:
            for re_entry in revert_entries:
                revert_amount = re_entry["mid"] - peak_price
                if revert_amount >= peak_move * REVERT_FRACTION:
                    reverted = True
                    revert_ts = re_entry["ts_wall"]
                    revert_speed = revert_ts - peak_ts
                    min_revert_price = re_entry["mid"]
                    break

        # Classify: is this near a known goal?
        is_goal = False
        nearest_goal_dist = float("inf")
        for gts in goal_timestamps:
            dist = abs(ts - gts)
            if dist < nearest_goal_dist:
                nearest_goal_dist = dist
            if dist < 120:
                is_goal = True

        # Estimate match minute
        match_minute = estimate_match_minute(ts, events, match_start_utc)

        # Compute entry prices at different delays
        entry_prices = {}
        for delay in ENTRY_DELAYS:
            entry_sample = sample_price_at(timeline, ts + delay)
            if entry_sample is not None:
                entry_prices[delay] = entry_sample["mid"]

        # Compute worst-case drawdown if we entered at each delay
        drawdowns = {}
        for delay in ENTRY_DELAYS:
            if delay not in entry_prices:
                continue
            ep = entry_prices[delay]

            if reverted and revert_ts is not None:
                # We detect the reversal EXIT_REACTION_TIME after it starts dropping
                # The reversal starts at peak_ts, we'd exit at peak_ts + reaction
                exit_sample = sample_price_at(
                    timeline, revert_ts + EXIT_REACTION_TIME
                )
                if exit_sample is not None:
                    exit_price = exit_sample["mid"]
                else:
                    # Fallback: use the reverted price
                    exit_price = min_revert_price if min_revert_price else baseline_mid

                # Drawdown = entry - exit (for "up" spike, we bought high)
                if spike_direction == "up":
                    loss_per_contract = (ep - exit_price) * 100  # cents
                else:
                    loss_per_contract = (exit_price - ep) * 100
                # Add spread costs (entry + exit)
                spread_at_entry = trajectory.get(delay, {}).get("spread")
                spread_cost = 0
                if spread_at_entry is not None:
                    spread_cost += spread_at_entry * 100 / 2  # half-spread on entry
                spread_cost += EXIT_SLIPPAGE_CENTS  # exit slippage

                drawdowns[delay] = {
                    "entry_price_c": round(ep * 100, 1),
                    "exit_price_c": round(exit_price * 100, 1),
                    "raw_loss_c": round(loss_per_contract, 1),
                    "spread_cost_c": round(spread_cost, 1),
                    "total_loss_c": round(loss_per_contract + spread_cost, 1),
                }

        spikes.append({
            "match_id": match_id,
            "ts_wall": spike_start_ts,
            "utc": _utc_str(spike_start_ts),
            "market": market_role,
            "direction": spike_direction,
            "baseline_c": round(baseline_mid * 100, 1),
            "peak_c": round(peak_price * 100, 1),
            "peak_move_c": round(peak_move * 100, 1),
            "time_to_peak_s": round(peak_ts - spike_start_ts, 1),
            "is_goal": is_goal,
            "reverted": reverted,
            "revert_time_s": round(revert_speed, 1) if revert_speed is not None else None,
            "match_minute": round(match_minute, 0) if match_minute is not None else None,
            "is_late_game": match_minute is not None and match_minute >= 75,
            "trajectory": trajectory,
            "entry_prices": entry_prices,
            "drawdowns": drawdowns,
        })

        # Skip forward 15s to avoid double-counting
        while i < len(timeline) and timeline[i]["ts_wall"] < ts + 15:
            i += 1
        continue

    return spikes


# ─── Dollar risk modeling ────────────────────────────────────────────


def compute_dollar_risk(
    drawdown_cents: float,
    contract_price_cents: float = CONTRACT_PRICE_CENTS,
    bankroll: float = BANKROLL,
    kelly_frac: float = HALF_KELLY,
) -> dict:
    """Compute dollar exposure and loss for a given drawdown."""
    position_dollars = bankroll * kelly_frac
    contracts = int(position_dollars / (contract_price_cents / 100))
    loss_dollars = contracts * (drawdown_cents / 100)
    loss_pct = (loss_dollars / bankroll) * 100 if bankroll > 0 else 0

    return {
        "position_dollars": round(position_dollars, 2),
        "contracts": contracts,
        "loss_dollars": round(loss_dollars, 2),
        "loss_pct_bankroll": round(loss_pct, 2),
    }


# ─── Reporting ───────────────────────────────────────────────────────


def format_report(
    all_spikes: list[dict],
    match_metas: dict[str, dict],
) -> str:
    lines = []

    # ── Section 1: All reversal events ──
    reversals = [s for s in all_spikes if s["reverted"] and not s["is_goal"]]
    goal_reversals = [s for s in all_spikes if s["reverted"] and s["is_goal"]]
    non_goal_spikes = [s for s in all_spikes if not s["is_goal"]]
    goal_spikes = [s for s in all_spikes if s["is_goal"]]

    lines.append("=" * 90)
    lines.append("SPIKE REVERSAL & DRAWDOWN ANALYSIS")
    lines.append(f"Total spikes >= {SPIKE_THRESHOLD*100:.0f}c: {len(all_spikes)}")
    lines.append(f"  Goal-related: {len(goal_spikes)}")
    lines.append(f"  Non-goal (false signals): {len(non_goal_spikes)}")
    lines.append(f"  Reversed (non-goal): {len(reversals)}")
    lines.append(f"  Reversed (goal — VAR/disallowed): {len(goal_reversals)}")
    lines.append("=" * 90)

    # ── Section 2: Detailed reversal events ──
    lines.append("")
    lines.append("REVERSAL EVENTS (spikes that reversed >=50%)")
    lines.append("-" * 90)
    lines.append(
        f"{'Match':<25} {'Time':>9} {'Min':>4} {'Base':>5} {'Peak':>5} "
        f"{'Move':>5} {'Revert':>7} {'Speed':>6} {'Goal?':>5}"
    )
    lines.append("-" * 90)

    all_reversals = sorted(
        [s for s in all_spikes if s["reverted"]],
        key=lambda s: s["peak_move_c"],
        reverse=True,
    )

    for s in all_reversals:
        meta = match_metas.get(s["match_id"], {})
        match_name = f"{meta.get('home_team', '?')[:10]}-{meta.get('away_team', '?')[:10]}"
        minute = f"{s['match_minute']:.0f}" if s["match_minute"] is not None else "?"
        revert_s = f"{s['revert_time_s']:.0f}s" if s["revert_time_s"] is not None else "?"
        goal_flag = "YES" if s["is_goal"] else "no"

        lines.append(
            f"{match_name:<25} {s['utc']:>9} {minute:>4} "
            f"{s['baseline_c']:>5.0f}c {s['peak_c']:>5.0f}c "
            f"{s['peak_move_c']:>4.0f}c {revert_s:>7} {goal_flag:>5}"
        )

    # ── Section 3: Worst-case drawdown per contract ──
    lines.append("")
    lines.append("=" * 90)
    lines.append("WORST-CASE DRAWDOWN PER CONTRACT")
    lines.append("(If SpikeDetector triggers and we enter, then spike fully reverts)")
    lines.append("-" * 90)

    # Focus on non-goal reversals (false positives) + goal reversals (VAR)
    danger_spikes = [s for s in all_spikes if s["reverted"] and s["drawdowns"]]

    if danger_spikes:
        lines.append(
            f"{'Match':<20} {'UTC':>9} {'Min':>4} {'Entry@':>6} "
            f"{'Entry':>6} {'Exit':>6} {'Raw':>5} {'Spread':>6} {'TOTAL':>6} {'Goal?':>5}"
        )
        lines.append("-" * 90)

        worst_cases = []
        for s in sorted(danger_spikes, key=lambda x: max(
            (d.get("total_loss_c", 0) for d in x["drawdowns"].values()), default=0
        ), reverse=True):
            meta = match_metas.get(s["match_id"], {})
            match_name = f"{meta.get('home_team', '?')[:8]}-{meta.get('away_team', '?')[:8]}"
            minute = f"{s['match_minute']:.0f}" if s["match_minute"] is not None else "?"

            for delay in ENTRY_DELAYS:
                dd = s["drawdowns"].get(delay)
                if dd is None:
                    continue
                worst_cases.append({
                    **s,
                    "delay": delay,
                    "dd": dd,
                    "match_name": match_name,
                    "minute": minute,
                })
                lines.append(
                    f"{match_name:<20} {s['utc']:>9} {minute:>4} "
                    f"+{delay}s{'':<2} "
                    f"{dd['entry_price_c']:>5.0f}c {dd['exit_price_c']:>5.0f}c "
                    f"{dd['raw_loss_c']:>4.0f}c {dd['spread_cost_c']:>5.1f}c "
                    f"{dd['total_loss_c']:>5.0f}c "
                    f"{'YES' if s['is_goal'] else 'no':>5}"
                )
    else:
        lines.append("  No reversal events with computable drawdowns found.")

    # ── Section 4: Late-game amplification ──
    lines.append("")
    lines.append("=" * 90)
    lines.append("LATE-GAME AMPLIFICATION (75min+ vs 0-75min)")
    lines.append("-" * 90)

    early_spikes = [s for s in all_spikes if s["match_minute"] is not None and s["match_minute"] < 75]
    late_spikes = [s for s in all_spikes if s.get("is_late_game", False)]

    def spike_stats(spike_list: list[dict], label: str) -> list[str]:
        stat_lines = []
        if not spike_list:
            stat_lines.append(f"  {label}: No spikes")
            return stat_lines

        moves = [s["peak_move_c"] for s in spike_list]
        reversals_in = [s for s in spike_list if s["reverted"]]
        rev_rate = len(reversals_in) / len(spike_list) * 100 if spike_list else 0

        stat_lines.append(f"  {label}:")
        stat_lines.append(f"    Count:            {len(spike_list)}")
        stat_lines.append(f"    Mean spike:       {statistics.mean(moves):.1f}c")
        stat_lines.append(f"    Median spike:     {statistics.median(moves):.1f}c")
        stat_lines.append(f"    Max spike:        {max(moves):.1f}c")
        stat_lines.append(f"    Reversal rate:    {rev_rate:.0f}% ({len(reversals_in)}/{len(spike_list)})")

        # Worst drawdown
        max_dd = 0
        for s in spike_list:
            for delay, dd in s.get("drawdowns", {}).items():
                if dd["total_loss_c"] > max_dd:
                    max_dd = dd["total_loss_c"]
        stat_lines.append(f"    Worst drawdown:   {max_dd:.1f}c/contract")

        return stat_lines

    lines.extend(spike_stats(early_spikes, "0-75 min"))
    lines.append("")
    lines.extend(spike_stats(late_spikes, "75+ min (late game)"))

    # ── Section 5: Dollar risk per match ──
    lines.append("")
    lines.append("=" * 90)
    lines.append("DOLLAR RISK PER MATCH (Kelly-sized positions)")
    lines.append(f"Bankroll: ${BANKROLL:.0f} | Half-Kelly (provisional): {HALF_KELLY*100:.0f}% | "
                 f"Full-Kelly: {KELLY_FRACTION*100:.0f}%")
    lines.append("-" * 90)

    # Compute for each sizing approach at worst-case drawdown
    all_drawdowns = []
    for s in all_spikes:
        for delay, dd in s.get("drawdowns", {}).items():
            all_drawdowns.append({
                **dd,
                "match_id": s["match_id"],
                "minute": s["match_minute"],
                "is_late": s.get("is_late_game", False),
                "is_goal": s["is_goal"],
                "reverted": s["reverted"],
            })

    reversal_drawdowns = [d for d in all_drawdowns if d["reverted"]]

    if reversal_drawdowns:
        worst_dd = max(reversal_drawdowns, key=lambda d: d["total_loss_c"])
        p50_dd = sorted(reversal_drawdowns, key=lambda d: d["total_loss_c"])[
            len(reversal_drawdowns) // 2
        ]
        mean_dd = statistics.mean(d["total_loss_c"] for d in reversal_drawdowns)

        for label, dd_c in [
            ("Median reversal", p50_dd["total_loss_c"]),
            ("Mean reversal", mean_dd),
            ("Worst-case reversal", worst_dd["total_loss_c"]),
        ]:
            for kelly_label, kelly_f in [
                ("Half-Kelly (proposed)", HALF_KELLY),
                ("Full-Kelly (no protection)", KELLY_FRACTION),
            ]:
                risk = compute_dollar_risk(dd_c, kelly_frac=kelly_f)
                lines.append(
                    f"  {label:<25} @ {kelly_label:<28}: "
                    f"${risk['loss_dollars']:>6.2f} loss "
                    f"({risk['loss_pct_bankroll']:.1f}% of bankroll, "
                    f"{risk['contracts']} contracts)"
                )
            lines.append("")

    # ── Section 6: Late-game worst case specifically ──
    late_reversal_dds = [d for d in reversal_drawdowns if d["is_late"]]
    if late_reversal_dds:
        lines.append("LATE-GAME (75min+) WORST CASE:")
        worst_late = max(late_reversal_dds, key=lambda d: d["total_loss_c"])
        for kelly_label, kelly_f in [
            ("Half-Kelly", HALF_KELLY),
            ("Full-Kelly", KELLY_FRACTION),
            ("Quarter-Kelly (proposed late-game cap)", HALF_KELLY / 2),
        ]:
            risk = compute_dollar_risk(worst_late["total_loss_c"], kelly_frac=kelly_f)
            lines.append(
                f"  {kelly_label:<40}: "
                f"${risk['loss_dollars']:>6.2f} loss "
                f"({risk['loss_pct_bankroll']:.1f}% of bankroll)"
            )
        lines.append("")

    # ── Section 7: Spike shape analysis ──
    lines.append("=" * 90)
    lines.append("SPIKE SHAPE ANALYSIS (real goals vs false signals vs VAR reversals)")
    lines.append("-" * 90)

    def shape_stats(spike_list: list[dict], label: str) -> list[str]:
        stat_lines = []
        if not spike_list:
            stat_lines.append(f"  {label}: No data")
            return stat_lines

        stat_lines.append(f"  {label} (n={len(spike_list)}):")

        # Time to peak
        ttp = [s["time_to_peak_s"] for s in spike_list]
        stat_lines.append(
            f"    Time to peak:   mean={statistics.mean(ttp):.1f}s, "
            f"median={statistics.median(ttp):.1f}s"
        )

        # Price at +1s, +3s, +5s as fraction of peak
        for offset in [1, 3, 5]:
            fracs = []
            for s in spike_list:
                t = s["trajectory"].get(offset)
                if t is not None and s["peak_move_c"] > 0:
                    adj = abs(t["mid"] * 100 - s["baseline_c"]) / s["peak_move_c"]
                    fracs.append(adj)
            if fracs:
                stat_lines.append(
                    f"    Adj@+{offset}s:       mean={statistics.mean(fracs)*100:.0f}%, "
                    f"median={statistics.median(fracs)*100:.0f}%"
                )

        # Spread at +1s
        spreads_1s = []
        for s in spike_list:
            t = s["trajectory"].get(1)
            if t is not None and t["spread"] is not None:
                spreads_1s.append(t["spread"] * 100)
        if spreads_1s:
            stat_lines.append(
                f"    Spread@+1s:     mean={statistics.mean(spreads_1s):.1f}c, "
                f"max={max(spreads_1s):.1f}c"
            )

        return stat_lines

    confirmed_goals = [s for s in all_spikes if s["is_goal"] and not s["reverted"]]
    var_reversals = [s for s in all_spikes if s["is_goal"] and s["reverted"]]
    false_signals_rev = [s for s in all_spikes if not s["is_goal"] and s["reverted"]]
    false_signals_sustained = [s for s in all_spikes if not s["is_goal"] and not s["reverted"]]

    lines.extend(shape_stats(confirmed_goals, "Confirmed goals"))
    lines.append("")
    lines.extend(shape_stats(var_reversals, "VAR-reversed goals"))
    lines.append("")
    lines.extend(shape_stats(false_signals_rev, "False signals (reversed)"))
    lines.append("")
    lines.extend(shape_stats(false_signals_sustained, "False signals (sustained — non-goal but price held)"))

    # ── Section 8: Risk control recommendations ──
    lines.append("")
    lines.append("=" * 90)
    lines.append("RISK CONTROL ANALYSIS")
    lines.append("=" * 90)

    # A. Position size caps by match minute
    lines.append("")
    lines.append("A. POSITION SIZE BY MATCH MINUTE")
    lines.append("-" * 60)

    minute_buckets = [(0, 30), (30, 60), (60, 75), (75, 90), (90, 120)]
    for lo, hi in minute_buckets:
        bucket_spikes = [
            s for s in all_spikes
            if s["match_minute"] is not None and lo <= s["match_minute"] < hi
        ]
        if bucket_spikes:
            moves = [s["peak_move_c"] for s in bucket_spikes]
            rev_count = sum(1 for s in bucket_spikes if s["reverted"])
            lines.append(
                f"  {lo}-{hi}min: {len(bucket_spikes)} spikes, "
                f"avg {statistics.mean(moves):.1f}c, max {max(moves):.1f}c, "
                f"{rev_count} reversed"
            )
        else:
            lines.append(f"  {lo}-{hi}min: no spikes")

    # B. Stop-loss analysis
    lines.append("")
    lines.append("B. STOP-LOSS EFFECTIVENESS")
    lines.append("-" * 60)
    lines.append("Would a 5c stop-loss (exit if price drops 5c from entry within 10s) help?")
    lines.append("")

    stop_would_help = 0
    stop_total = 0
    stop_saved_cents = []
    for s in all_spikes:
        if not s["reverted"]:
            continue
        for delay in ENTRY_DELAYS[:2]:  # focus on +1s and +2s entry
            dd = s["drawdowns"].get(delay)
            if dd is None:
                continue
            stop_total += 1
            # If raw loss > 5c, stop would have triggered
            if dd["raw_loss_c"] > 5:
                stop_would_help += 1
                saved = dd["raw_loss_c"] - 5 - EXIT_SLIPPAGE_CENTS
                if saved > 0:
                    stop_saved_cents.append(saved)

    if stop_total > 0:
        lines.append(
            f"  Reversals where stop triggers: {stop_would_help}/{stop_total} "
            f"({stop_would_help/stop_total*100:.0f}%)"
        )
        if stop_saved_cents:
            lines.append(
                f"  Average saved per trigger:     {statistics.mean(stop_saved_cents):.1f}c/contract"
            )
    else:
        lines.append("  No reversal data to analyze stop-loss effectiveness.")

    # C. Multi-contract confirmation
    lines.append("")
    lines.append("C. MULTI-CONTRACT CONFIRMATION POTENTIAL")
    lines.append("-" * 60)
    lines.append(
        "Note: Current analysis only tracks home_win market.\n"
        "Multi-contract confirmation (home_win UP + away_win DOWN simultaneously)\n"
        "requires extending the spike detector to monitor all 3 markets.\n"
        "This is the highest-ROI improvement for reducing false positives."
    )

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────


def main() -> None:
    if not LATENCY_DIR.exists():
        print(f"ERROR: {LATENCY_DIR} does not exist")
        sys.exit(1)

    match_dirs = sorted(
        p for p in LATENCY_DIR.iterdir()
        if p.is_dir() and (p / "kalshi.jsonl").exists()
    )

    print("=" * 70)
    print("SPIKE REVERSAL & DRAWDOWN ANALYSIS")
    print(f"Matches: {len(match_dirs)}")
    print(f"Spike threshold: {SPIKE_THRESHOLD*100:.0f}c")
    print("=" * 70)

    all_spikes: list[dict] = []
    match_metas: dict[str, dict] = {}

    for match_dir in match_dirs:
        meta = load_metadata(match_dir)
        match_id = meta.get("match_id", match_dir.name)
        match_metas[match_id] = meta
        home = meta.get("home_team", "?")
        away = meta.get("away_team", "?")

        print(f"\n--- {match_id}: {home} vs {away} ---")

        events = load_events(match_dir)
        ticker_roles = identify_ticker_roles(meta)

        # Get goal timestamps for classification
        goal_timestamps = [
            e["ts_wall"] for e in events if e["type"] == "goal"
        ]

        print(f"  Goals: {len(goal_timestamps)}, loading orderbooks...")

        market_timelines = build_all_orderbook_timelines(match_dir, ticker_roles)

        for role in ["home", "away"]:
            tl = market_timelines.get(role, [])
            if not tl:
                continue

            print(f"  Scanning {role} market ({len(tl)} entries)...")
            spikes = find_all_spikes(
                tl, events, goal_timestamps,
                match_id, f"{role}_win",
                meta.get("started_utc", ""),
            )
            all_spikes.extend(spikes)
            n_rev = sum(1 for s in spikes if s["reverted"])
            print(f"    Found {len(spikes)} spikes >= {SPIKE_THRESHOLD*100:.0f}c, "
                  f"{n_rev} reversed")

    # ── Generate report ──
    report = format_report(all_spikes, match_metas)
    print("\n" + report)

    # ── Save outputs ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw JSON
    json_path = OUTPUT_DIR / "spike_reversals.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "spike_threshold_c": SPIKE_THRESHOLD * 100,
                "revert_window_s": REVERT_WINDOW,
                "revert_fraction": REVERT_FRACTION,
                "entry_delays_s": ENTRY_DELAYS,
                "exit_reaction_s": EXIT_REACTION_TIME,
                "exit_slippage_c": EXIT_SLIPPAGE_CENTS,
                "bankroll": BANKROLL,
                "half_kelly": HALF_KELLY,
                "full_kelly": KELLY_FRACTION,
            },
            "summary": {
                "total_spikes": len(all_spikes),
                "goal_spikes": sum(1 for s in all_spikes if s["is_goal"]),
                "false_spikes": sum(1 for s in all_spikes if not s["is_goal"]),
                "reversed": sum(1 for s in all_spikes if s["reverted"]),
                "late_game_spikes": sum(1 for s in all_spikes if s.get("is_late_game")),
            },
            "spikes": all_spikes,
        }, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Report text
    report_path = OUTPUT_DIR / "reversal_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
