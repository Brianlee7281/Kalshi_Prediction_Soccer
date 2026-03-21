"""Kalshi live data poller — polls match state every 1s and dispatches events.

Replaces goalserve_poller as the Phase 3 event source when Kalshi live data
is available. Produces identical downstream effects on LiveMatchModel.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from src.clients.kalshi_live_data import KalshiLiveDataClient, MatchState
from src.common.logging import get_logger
from src.engine.event_handlers import (
    handle_goal,
    handle_period_change,
    handle_red_card,
)

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.kalshi_live_poller")

_POLL_INTERVAL_S = 1.0  # well within Basic tier 20 reads/s


async def kalshi_live_poller(
    model: LiveMatchModel,
    client: KalshiLiveDataClient | None = None,
    poll_interval: float = _POLL_INTERVAL_S,
    replay_mode: bool = False,
) -> None:
    """Coroutine: poll Kalshi live data every poll_interval seconds.

    Runs until model.engine_phase == "FINISHED".
    Records all poll responses to JSONL if recorder is attached.

    Args:
        model: Shared live match state.
        client: Optional pre-configured client (for replay mode).
                If None, creates a live client from env vars.
        poll_interval: Seconds between polls (default 1.0, use 1.0/speed for replay).
        replay_mode: If True, drives model.t from replay data instead of wall clock.
    """
    if client is None:
        api_key = os.environ["KALSHI_API_KEY"]
        private_key_path = os.environ["KALSHI_PRIVATE_KEY_PATH"]
        client = KalshiLiveDataClient(api_key=api_key, private_key_path=private_key_path)

    try:
        milestone_uuid = await client.resolve_milestone_uuid(model.kalshi_event_ticker)
    except Exception as exc:
        raise RuntimeError(
            f"kalshi_live_poller: failed to resolve milestone for "
            f"event_ticker={model.kalshi_event_ticker!r}: {exc}"
        ) from exc

    logger.info(
        "kalshi_milestone_resolved",
        match_id=model.match_id,
        event_ticker=model.kalshi_event_ticker,
        milestone_uuid=milestone_uuid,
    )

    first_poll = True
    next_tick = time.monotonic()

    try:
        while model.engine_phase != "FINISHED":
            next_tick += poll_interval

            try:
                state = await asyncio.wait_for(
                    client.get_live_data(milestone_uuid),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("kalshi_live_poll_error", error=str(exc))
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                continue

            # Recording
            recorder = getattr(model, "recorder", None)
            if recorder is not None:
                recorder.record_kalshi_live_data(state.model_dump())

            # Late join time sync (first poll with live status)
            if first_poll and state.status == "live":
                now = time.monotonic()
                model.kickoff_wall_clock = now - ((state.minute + state.stoppage) * 60.0)
                model.t = float(state.minute + state.stoppage)
                logger.info(
                    "late_join_time_sync",
                    match_id=model.match_id,
                    minute=state.minute,
                    stoppage=state.stoppage,
                )
                first_poll = False
            if first_poll and state.half in ("HT", "FT"):
                first_poll = False

            # Replay mode: drive model.t from replay data (not wall clock)
            if replay_mode and state.status == "live":
                model.t = float(state.minute + state.stoppage)

            # Event detection and dispatch
            events = _detect_events_from_state(model, state)
            for event in events:
                etype = event["type"]
                if etype == "goal":
                    handle_goal(model, event["team"], event["minute"])
                elif etype == "red_card":
                    handle_red_card(model, event["team"], event["minute"])
                elif etype == "period_change":
                    handle_period_change(model, event["new_phase"])

                if recorder is not None:
                    recorder.record_event({
                        **event,
                        "score_after": list(model.score),
                        "engine_phase": model.engine_phase,
                        "model_t": model.t,
                    })

            # Stoppage time update (mirrors goalserve_poller logic)
            if (
                state.stoppage > 0
                and model.engine_phase == "SECOND_HALF"
                and model.t >= 85.0
            ):
                model.update_T_exp(state.stoppage)

            # HMM/DomIndex: no shots/corners/possession from Kalshi — pass None.
            # Goals are fed via handle_goal → hmm_estimator.record_goal().
            # update(None, t) keeps _current_t fresh so decay is correct.
            if model.hmm_estimator is not None:
                model.hmm_estimator.update(None, model.t)

            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    finally:
        await client.close()

    logger.info("kalshi_live_poller_finished", match_id=model.match_id)


def _detect_events_from_state(
    model: LiveMatchModel,
    state: MatchState,
) -> list[dict]:
    """Compare MatchState with model state, detect all events.

    Returns list of event dicts: [{type: "goal", team: "home", minute: 35}, ...]
    Handles multi-goal: if score jumped by 2+, emits sequential goals.
    """
    events: list[dict] = []

    # ── Score diff → goals ────────────────────────────────────
    prev_home, prev_away = model._last_score
    home_diff = state.home_score - prev_home
    away_diff = state.away_score - prev_away

    for i in range(home_diff):
        events.append({
            "type": "goal",
            "team": "home",
            "minute": state.minute,
            "t": model.t + i * 0.1,
            "occurence_ts": state.last_play_ts,
        })

    for i in range(away_diff):
        events.append({
            "type": "goal",
            "team": "away",
            "minute": state.minute,
            "t": model.t + (home_diff + i) * 0.1,
            "occurence_ts": state.last_play_ts,
        })

    model._last_score = (state.home_score, state.away_score)

    # ── Period change ─────────────────────────────────────────
    new_phase = _kalshi_half_to_phase(state.half)
    if new_phase is not None and new_phase != model._last_period:
        events.append({"type": "period_change", "new_phase": new_phase})

    # ── Red cards from significant_events ─────────────────────
    processed: set[str] | None = getattr(model, "_processed_red_cards", None)
    if processed is None:
        processed = set()
        model._processed_red_cards = processed  # type: ignore[attr-defined]

    for ev in state.significant_events:
        if ev.get("event_type") != "red_card":
            continue
        team = ev.get("team", "")
        player = ev.get("player", "")
        ev_time = ev.get("time", "")
        dedup_key = f"{team}_{player}"
        if dedup_key in processed:
            continue
        processed.add(dedup_key)
        try:
            minute = int(str(ev_time).rstrip("'"))
        except (ValueError, TypeError):
            minute = state.minute
        events.append({
            "type": "red_card",
            "team": team,
            "minute": minute,
        })

    return events


def _kalshi_half_to_phase(half: str) -> str | None:
    """Map Kalshi half string to engine phase.

    "1st" → "FIRST_HALF"
    "2nd" → "SECOND_HALF"
    "HT"  → "HALFTIME"
    "FT"  → "FINISHED"
    else  → None
    """
    _MAP: dict[str, str] = {
        "1st": "FIRST_HALF",
        "2nd": "SECOND_HALF",
        "HT": "HALFTIME",
        "FT": "FINISHED",
    }
    return _MAP.get(half)
