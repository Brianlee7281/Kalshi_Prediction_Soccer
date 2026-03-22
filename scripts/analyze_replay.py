"""Analyze replay results — P_model timeline + goal impact.

Usage:
  PYTHONPATH=. python scripts/analyze_replay.py data/replay_results/KXEPLGAME-26MAR20BOUMUN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_replay.py <replay_results_dir>")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    ticks = load_jsonl(result_dir / "ticks.jsonl")
    events = load_jsonl(result_dir / "events.jsonl")

    # Load original metadata for actual result
    match_id = result_dir.name
    orig_meta_path = Path("data/recordings") / match_id / "metadata.json"
    actual_score = None
    home_team = "Home"
    away_team = "Away"
    if orig_meta_path.exists():
        with open(orig_meta_path) as f:
            orig = json.load(f)
            actual_score = orig.get("final_score")
            home_team = orig.get("home_team", "Home")
            away_team = orig.get("away_team", "Away")

    # ── Extract tick data ──────────────────────────────────
    t_arr = [tick["t"] for tick in ticks]
    hw = [tick["P_model"]["home_win"] for tick in ticks]
    dr = [tick["P_model"]["draw"] for tick in ticks]
    aw = [tick["P_model"]["away_win"] for tick in ticks]

    # ── Extract events ─────────────────────────────────────
    goals = [e for e in events if e["type"] == "goal"]
    red_cards = [e for e in events if e["type"] == "red_card"]

    # ── Print summary ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {home_team} vs {away_team}")
    if actual_score:
        print(f"  Final: {actual_score[0]} - {actual_score[1]}")
    print(f"  Ticks: {len(ticks)}  |  Goals: {len(goals)}  |  Red Cards: {len(red_cards)}")
    print(f"{'='*65}")

    # ── Timeline table: show tick before and after each score change ──
    print(f"\n  P_model Timeline (key moments):")
    print(f"  {'t':>5}  {'Score':>7}  {'P(Home)':>8}  {'P(Draw)':>8}  {'P(Away)':>8}  Event")
    print(f"  {'-'*60}")

    # Find score-change boundaries in ticks
    prev_score = None
    key_indices = [0]  # always show first tick
    for i, tick in enumerate(ticks):
        score = tuple(tick["score"])
        if prev_score is not None and score != prev_score:
            # Show tick before score change (last with old score)
            if i - 1 not in key_indices:
                key_indices.append(i - 1)
            # Show tick after score change (first with new score)
            key_indices.append(i)
        prev_score = score
    key_indices.append(len(ticks) - 1)  # always show last tick

    for i in sorted(set(key_indices)):
        tick = ticks[i]
        t = tick["t"]
        s = tick["score"]
        p = tick["P_model"]

        # Find matching event
        event_str = ""
        for g in goals:
            if g["score_after"] == s and abs(g["minute"] - t) <= 1:
                team_name = home_team if g["team"] == "home" else away_team
                event_str = f"<< GOAL ({team_name})"
                break

        print(f"  {t:5.0f}  {s[0]:>2}-{s[1]:<2}    {p['home_win']:7.4f}   {p['draw']:7.4f}   {p['away_win']:7.4f}   {event_str}")

    # ── Goal impact analysis (compare tick before vs after score change) ──
    if goals:
        print(f"\n  Goal Impact (P_model change at score change):")
        print(f"  {'Min':>5}  {'Team':>12}  {'Score':>7}  {'dP(Home)':>9}  {'dP(Draw)':>9}  {'dP(Away)':>9}")
        print(f"  {'-'*60}")

        prev_score = None
        prev_tick = None
        for tick in ticks:
            score = tuple(tick["score"])
            if prev_score is not None and score != prev_score and prev_tick is not None:
                # Find which goal caused this score change
                goal_team = ""
                goal_minute = tick["t"]
                for g in goals:
                    if g["score_after"] == list(score):
                        goal_team = g["team"]
                        goal_minute = g["minute"]
                        break

                team_name = home_team if goal_team == "home" else away_team
                dh = tick["P_model"]["home_win"] - prev_tick["P_model"]["home_win"]
                dd = tick["P_model"]["draw"] - prev_tick["P_model"]["draw"]
                da = tick["P_model"]["away_win"] - prev_tick["P_model"]["away_win"]
                print(f"  {goal_minute:5.0f}  {team_name:>12}  {score[0]:>2}-{score[1]:<2}   {dh:+8.4f}   {dd:+8.4f}   {da:+8.4f}")

            prev_score = score
            prev_tick = tick

    # ── Warnings ──────────────────────────────────────────
    print(f"\n  Diagnostics:")
    p0 = ticks[0]["P_model"]
    if p0["over_25"] >= 0.99:
        print(f"  [!] over_2.5 = {p0['over_25']:.2f} at kickoff (should be ~0.5) — mock params issue")
    if p0["draw"] < 0.05:
        print(f"  [!] P(Draw) = {p0['draw']:.4f} at kickoff (should be ~0.25) — mock params issue")
    sum_p = p0["home_win"] + p0["draw"] + p0["away_win"]
    print(f"  P(H)+P(D)+P(A) at kickoff = {sum_p:.4f} (should be 1.0)")
    print(f"  NOTE: Check params if P_model values seem unrealistic at kickoff.")

    # ── Plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(t_arr, hw, label=f"P({home_team} Win)", color="#2196F3", linewidth=2)
    ax.plot(t_arr, dr, label="P(Draw)", color="#4CAF50", linewidth=2)
    ax.plot(t_arr, aw, label=f"P({away_team} Win)", color="#F44336", linewidth=2)

    # Mark goals with score labels
    for g in goals:
        color = "#2196F3" if g["team"] == "home" else "#F44336"
        sa = g["score_after"]
        label = f"{sa[0]}-{sa[1]}"
        ax.axvline(x=g["minute"], color=color, linestyle="--", alpha=0.5)
        ax.annotate(
            label,
            xy=(g["minute"], 0.92),
            fontsize=9,
            fontweight="bold",
            ha="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8),
        )

    # Mark red cards
    for rc in red_cards:
        ax.axvline(x=rc["minute"], color="black", linestyle=":", alpha=0.4)
        ax.annotate("RC", xy=(rc["minute"], 0.02), fontsize=8, color="red", ha="center")

    # Halftime line
    ax.axvline(x=45, color="gray", linestyle="-", alpha=0.3, linewidth=2)
    ax.annotate("HT", xy=(45, 0.02), fontsize=9, color="gray", ha="center")

    title = f"{home_team} vs {away_team}"
    if actual_score:
        title += f"  (Final: {actual_score[0]}-{actual_score[1]})"
    ax.set_xlabel("Match Time (minutes)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(t_arr) if t_arr else 90)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    output_path = result_dir / "p_model_timeline.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {output_path}\n")


if __name__ == "__main__":
    main()
