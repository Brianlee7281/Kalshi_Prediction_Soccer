"""Step 1.1 — Interval Segmentation.

Converts parsed match data into IntervalRecord lists for math core consumption.
Each interval has constant state_X and delta_S.
"""
from __future__ import annotations

import structlog

from src.common.types import IntervalRecord, RedCardTransition

log = structlog.get_logger(__name__)

# Basis boundaries (finer resolution after 75')
_BASIS_BOUNDARIES = [0, 15, 30, 45, 60, 75, 85, 90]


def _compute_state_transition(state_X: int, team: str) -> int:
    """Compute new Markov state after a red card.

    State space: 0=11v11, 1=10v11, 2=11v10, 3=10v10
    Home red: 0→1, 2→3
    Away red: 0→2, 1→3
    """
    if team == "home":
        if state_X == 0:
            return 1
        if state_X == 2:
            return 3
    elif team == "away":
        if state_X == 0:
            return 2
        if state_X == 1:
            return 3
    return state_X


def segment_match_to_intervals(match: dict) -> list[IntervalRecord]:
    """Convert a single parsed match into IntervalRecords.

    Creates intervals split at:
    - Goal events (score change → new ΔS)
    - Red card events (state change → new X)
    - Halftime (45min → halftime interval → second half)
    - 15-min basis boundaries (0, 15, 30, 45, 60, 75, 90)

    Each interval has constant state_X and delta_S.
    """
    match_id = match["match_id"]

    # Collect all events with their times, sorted
    events: list[tuple[float, str, dict]] = []

    for g in match.get("goal_events", []):
        minute = float(g["minute"])
        events.append((minute, "goal", g))

    for rc in match.get("red_card_events", []):
        minute = float(rc["minute"])
        events.append((minute, "red_card", rc))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "red_card" else 1))

    # Collect all split points: events + halftime + basis boundaries
    split_times: set[float] = set()
    for t, _, _ in events:
        split_times.add(t)

    # Add basis boundaries
    for b in _BASIS_BOUNDARIES:
        split_times.add(float(b))

    # Add halftime at 45
    split_times.add(45.0)

    # Add match end at 90 (or later if stoppage)
    max_event_time = max((e[0] for e in events), default=90.0)
    match_end = max(90.0, max_event_time + 1.0)
    split_times.add(match_end)

    # Sort split points
    sorted_splits = sorted(split_times)

    # Walk through time, tracking state
    state_X = 0
    home_score = 0
    away_score = 0

    # Build event lookup by time
    events_at: dict[float, list[tuple[str, dict]]] = {}
    for t, etype, edata in events:
        events_at.setdefault(t, []).append((etype, edata))

    intervals: list[IntervalRecord] = []
    prev_t = 0.0

    for split_t in sorted_splits:
        if split_t <= prev_t:
            continue

        # Check if this segment crosses halftime
        if prev_t < 45.0 <= split_t:
            # First half portion up to 45
            if prev_t < 45.0:
                iv = IntervalRecord(
                    match_id=match_id,
                    t_start=prev_t,
                    t_end=45.0,
                    state_X=state_X,
                    delta_S=home_score - away_score,
                    is_halftime=False,
                )
                if iv.t_end > iv.t_start:
                    intervals.append(iv)

            # Halftime interval
            intervals.append(IntervalRecord(
                match_id=match_id,
                t_start=45.0,
                t_end=45.0,
                state_X=state_X,
                delta_S=home_score - away_score,
                is_halftime=True,
            ))

            prev_t = 45.0
            if split_t == 45.0:
                continue

        # Process events at split_t
        home_goals_this: list[float] = []
        away_goals_this: list[float] = []
        goal_delta_before: list[int] = []
        rc_transitions: list[RedCardTransition] = []

        for etype, edata in events_at.get(split_t, []):
            if etype == "red_card":
                from_state = state_X
                to_state = _compute_state_transition(state_X, edata["team"])
                rc_transitions.append(RedCardTransition(
                    minute=split_t,
                    from_state=from_state,
                    to_state=to_state,
                    team=edata["team"],
                ))
                state_X = to_state
            elif etype == "goal":
                goal_delta_before.append(home_score - away_score)
                if edata["team"] == "home":
                    home_goals_this.append(split_t)
                    home_score += 1
                else:
                    away_goals_this.append(split_t)
                    away_score += 1

        # Create interval from prev_t to split_t
        delta_S_before = (home_score - away_score)
        # delta_S for the interval is the state BEFORE events at split_t
        # But events happen at split_t, so the interval [prev_t, split_t) has
        # the state before the events
        iv_delta_S = (home_score - len(home_goals_this)) - (away_score - len(away_goals_this))

        iv = IntervalRecord(
            match_id=match_id,
            t_start=prev_t,
            t_end=split_t,
            state_X=state_X if not rc_transitions else rc_transitions[0].from_state,
            delta_S=iv_delta_S,
            is_halftime=False,
            home_goal_times=home_goals_this,
            away_goal_times=away_goals_this,
            goal_delta_before=goal_delta_before,
            red_card_transitions=rc_transitions,
        )

        if iv.t_end > iv.t_start:
            intervals.append(iv)

        prev_t = split_t

    return intervals


def segment_all_matches(matches: list[dict]) -> dict[str, list[IntervalRecord]]:
    """Segment all matches. Returns {match_id: [IntervalRecord, ...]}.

    Skips matches with parsing errors (log warning, don't crash).
    """
    result: dict[str, list[IntervalRecord]] = {}

    for match in matches:
        match_id = match.get("match_id", "unknown")
        try:
            intervals = segment_match_to_intervals(match)
            if intervals:
                result[match_id] = intervals
        except Exception as e:
            log.warning("segmentation_error", match_id=match_id, error=str(e))
            continue

    log.info("segmentation_complete", matches_segmented=len(result), matches_input=len(matches))
    return result
