"""Event handlers for goal, red card, and period change events.

Called by the Goalserve poller when events are detected. Each handler
mutates the shared LiveMatchModel in place.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.engine.intensity import compute_lambda

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.event_handlers")

# Cooldown durations (in ticks)
_GOAL_COOLDOWN_TICKS = 50
_RED_CARD_COOLDOWN_TICKS = 30

# Markov state transitions for red cards
# State encoding: 0=11v11, 1=10v11(home red), 2=11v10(away red), 3=10v10
_HOME_RED_TRANSITIONS = {0: 1, 2: 3}
_AWAY_RED_TRANSITIONS = {0: 2, 1: 3}


def handle_goal(
    model: LiveMatchModel,
    team: str,  # "home" | "away"
    minute: int,
) -> None:
    """Process a goal event.

    - Update score, delta_S
    - Set event_state = CONFIRMED
    - Set cooldown = True, cooldown_until_tick = tick_count + 50
    """
    home, away = model.score
    if team == "home":
        home += 1
    else:
        away += 1

    model.score = (home, away)
    model.delta_S = home - away
    model.event_state = "CONFIRMED"
    model.cooldown = True
    model.cooldown_until_tick = model.tick_count + _GOAL_COOLDOWN_TICKS

    logger.info(
        "goal_handled",
        match_id=model.match_id,
        team=team,
        minute=minute,
        score=model.score,
        delta_S=model.delta_S,
    )

    if model.hmm_estimator is not None:
        model.hmm_estimator.record_goal(model.t, team)

    if model.strength_updater is not None:
        # Compute current intensities so the v5 EKF goal-update path fires
        # (without these kwargs, update_on_goal falls back to v4 Bayesian).
        lambda_H = compute_lambda(model, "home")
        lambda_A = compute_lambda(model, "away")
        new_a_H, new_a_A = model.strength_updater.update_on_goal(
            team, model.mu_H_elapsed, model.mu_A_elapsed,
            lambda_H=lambda_H, lambda_A=lambda_A,
            dt=1.0 / 60.0,  # one-tick observation window, in minutes
        )
        model.a_H = new_a_H
        model.a_A = new_a_A
        model.last_goal_type = model.strength_updater.classify_goal(team).label


def handle_red_card(
    model: LiveMatchModel,
    team: str,  # "home" | "away"
    minute: int,
) -> None:
    """Process a red card event.

    - Update current_state_X (Markov transition)
    - State transitions: home red -> 0->1 or 2->3, away red -> 0->2 or 1->3
    - Set cooldown
    """
    transitions = _HOME_RED_TRANSITIONS if team == "home" else _AWAY_RED_TRANSITIONS
    new_state = transitions.get(model.current_state_X)

    if new_state is not None:
        model.current_state_X = new_state
    else:
        logger.warning(
            "red_card_no_transition",
            match_id=model.match_id,
            team=team,
            current_state=model.current_state_X,
        )

    model.cooldown = True
    model.cooldown_until_tick = model.tick_count + _RED_CARD_COOLDOWN_TICKS

    logger.info(
        "red_card_handled",
        match_id=model.match_id,
        team=team,
        minute=minute,
        state_X=model.current_state_X,
    )


def handle_period_change(
    model: LiveMatchModel,
    new_phase: str,
) -> None:
    """Process period change.

    - Spam prevention: only process if new_phase != _last_period
    - FIRST_HALF -> HALFTIME: record halftime_start
    - HALFTIME -> SECOND_HALF: compute halftime_accumulated
    - SECOND_HALF -> FINISHED: set engine_phase
    """
    if new_phase == model._last_period:
        return

    now = time.monotonic()

    if new_phase == "HALFTIME":
        model.halftime_start = now

    elif new_phase == "SECOND_HALF" and model.halftime_start > 0:
        model.halftime_accumulated = now - model.halftime_start

    model.engine_phase = new_phase
    model._last_period = new_phase

    logger.info(
        "period_change",
        match_id=model.match_id,
        new_phase=new_phase,
        halftime_accumulated=model.halftime_accumulated,
    )


def detect_events_from_poll(
    model: LiveMatchModel,
    poll_data: dict,
) -> list[dict]:
    """Compare poll data with model state, detect all events.

    Returns list of event dicts: [{type: "goal", team: "home", minute: 35}, ...]
    Handles multi-goal: if score jumped by 2+, emit sequential goals.
    """
    events: list[dict] = []

    # Extract current score from poll — Goalserve returns "?" before kickoff
    try:
        local_goals = int(poll_data.get("localteam", {}).get("@goals", "0"))
    except (ValueError, TypeError):
        local_goals = 0
    try:
        visitor_goals = int(poll_data.get("visitorteam", {}).get("@goals", "0"))
    except (ValueError, TypeError):
        visitor_goals = 0

    prev_home, prev_away = model._last_score

    # Detect home goals
    home_diff = local_goals - prev_home
    for i in range(home_diff):
        events.append({
            "type": "goal",
            "team": "home",
            "minute": int(model.t) if i == 0 else int(model.t),
            "t": model.t + i * 0.1,
        })

    # Detect away goals
    away_diff = visitor_goals - prev_away
    for i in range(away_diff):
        events.append({
            "type": "goal",
            "team": "away",
            "minute": int(model.t) if i == 0 else int(model.t),
            "t": model.t + (home_diff + i) * 0.1,
        })

    # Detect period change from status
    status = poll_data.get("@status", "")
    new_phase = _status_to_phase(status)
    if new_phase is not None and new_phase != model._last_period:
        events.append({
            "type": "period_change",
            "new_phase": new_phase,
        })

    # Detect red cards from events/incidents if present
    events.extend(_detect_red_cards(poll_data, model))

    # Update last known score
    model._last_score = (local_goals, visitor_goals)

    return events


def _status_to_phase(status: str) -> str | None:
    """Convert Goalserve status string to engine phase.

    Numeric string = in play (first or second half based on value).
    "HT" = halftime. "FT" = finished.
    """
    status = status.strip()
    if status == "HT":
        return "HALFTIME"
    if status == "FT":
        return "FINISHED"
    if status.isdigit():
        minute = int(status)
        if minute <= 45:
            return "FIRST_HALF"
        return "SECOND_HALF"
    return None


def _detect_red_cards(poll_data: dict, model: LiveMatchModel) -> list[dict]:
    """Detect red card events from Goalserve poll data.

    Checks the events/incidents section for new red cards not yet processed.
    """
    events: list[dict] = []
    incidents = poll_data.get("events", {})
    if not incidents:
        return events

    cards = incidents.get("redcard", [])
    if isinstance(cards, dict):
        cards = [cards]

    for card in cards:
        team = card.get("team", "").lower()
        if team not in ("home", "away", "localteam", "visitorteam"):
            continue
        # Normalize team names
        if team in ("localteam", "home"):
            team = "home"
        else:
            team = "away"

        minute_str = card.get("minute", "0")
        try:
            minute = int(minute_str)
        except (ValueError, TypeError):
            minute = int(model.t)

        events.append({
            "type": "red_card",
            "team": team,
            "minute": minute,
        })

    return events


def handle_penalty(
    model: LiveMatchModel,
    team: str,
    minute: int,
) -> None:
    """Process a penalty event — freeze orderbook until resolved."""
    model.event_state = "PENALTY_PENDING"
    model.ob_freeze = True
    logger.info("penalty_detected", match_id=model.match_id, team=team, minute=minute)


def handle_var_review(
    model: LiveMatchModel,
    minute: int,
) -> None:
    """Process a VAR review event — freeze orderbook until resolved."""
    model.event_state = "VAR_REVIEW"
    model.ob_freeze = True
    logger.info("var_review_started", match_id=model.match_id, minute=minute)
