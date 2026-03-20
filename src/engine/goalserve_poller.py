# DISABLED Sprint KLD-5: replaced by kalshi_live_poller. Kept for reference.
"""Goalserve poller — polls live scores every 3s and dispatches events.

Runs as a coroutine alongside tick_loop and odds_api_listener. Detects
goals, red cards, and period changes by comparing poll responses with
the current LiveMatchModel state.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from src.clients.goalserve import GoalserveClient
from src.common.logging import get_logger
from src.engine.event_handlers import (
    detect_events_from_poll,
    handle_goal,
    handle_period_change,
    handle_red_card,
)

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.goalserve_poller")

_POLL_INTERVAL_S = 3.0


async def goalserve_poller(model: LiveMatchModel) -> None:
    """Coroutine: poll Goalserve every 3s, detect events, update model state.

    Runs until model.engine_phase == "FINISHED".
    Records all poll responses to JSONL if recorder is attached.
    """
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        logger.error("goalserve_poller_no_key")
        return

    client = GoalserveClient(api_key)
    first_poll = True

    # Shared lock prevents double-firing when both pollers run in parallel
    event_lock: asyncio.Lock | None = getattr(model, "_event_lock", None)
    if event_lock is None:
        event_lock = asyncio.Lock()
        model._event_lock = event_lock  # type: ignore[attr-defined]

    try:
        while model.engine_phase != "FINISHED":
            try:
                live_data = await asyncio.wait_for(
                    client.get_live_scores(), timeout=10.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("goalserve_poll_error", error=str(exc))
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue

            # Find our match — search @id, @fix_id, @static_id
            match_data = client.find_match_in_live(model.match_id, live_data)

            if match_data is None:
                logger.debug("goalserve_match_not_found", match_id=model.match_id)
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue

            # Record raw poll response
            recorder = getattr(model, "recorder", None)
            if recorder is not None:
                recorder.record_goalserve(match_data)

            # Late container join: first poll with numeric status → sync time
            status = match_data.get("@status", "")
            if first_poll and status.isdigit():
                minute = int(status)
                now = time.monotonic()
                model.kickoff_wall_clock = now - (minute * 60.0)
                model.t = float(minute)
                logger.info(
                    "late_join_time_sync",
                    match_id=model.match_id,
                    minute=minute,
                )
                first_poll = False

            if first_poll and status in ("HT", "FT") or status.isdigit():
                first_poll = False

            # Detect + dispatch under lock to prevent double-firing with kalshi_live_poller
            async with event_lock:
                events = detect_events_from_poll(model, match_data)
                for event in events:
                    etype = event["type"]
                    if etype == "goal":
                        logger.info(
                            "goal_source",
                            source="goalserve",
                            team=event["team"],
                            minute=event["minute"],
                            match_id=model.match_id,
                        )
                        handle_goal(model, event["team"], event["minute"])
                    elif etype == "red_card":
                        handle_red_card(model, event["team"], event["minute"])
                    elif etype == "period_change":
                        handle_period_change(model, event["new_phase"])

            # v5: Extract live_stats for Layer 2 HMM/DomIndex
            live_stats = _extract_live_stats(match_data)
            if live_stats is not None:
                hmm = getattr(model, "hmm_estimator", None)
                if hmm is not None:
                    hmm.update(live_stats, model.t)

            # Check for stoppage time announcement (second half, minute >= 85)
            inj_str = match_data.get("@inj_minute", "")
            if (
                inj_str
                and inj_str.isdigit()
                and model.engine_phase == "SECOND_HALF"
                and model.t >= 85.0
            ):
                inj_minute = int(inj_str)
                if inj_minute > 0:
                    model.update_T_exp(inj_minute)

            await asyncio.sleep(_POLL_INTERVAL_S)

    finally:
        await client.close()

    logger.info("goalserve_poller_finished", match_id=model.match_id)


def _extract_live_stats(match_data: dict) -> dict | None:
    """Extract live statistics from Goalserve poll for Layer 2."""
    stats = match_data.get("stats", {})
    if not stats:
        return None
    try:
        return {
            "shots_on_target_h": int(stats.get("shotsontarget", {}).get("localteam", 0)),
            "shots_on_target_a": int(stats.get("shotsontarget", {}).get("visitorteam", 0)),
            "corners_h": int(stats.get("corners", {}).get("localteam", 0)),
            "corners_a": int(stats.get("corners", {}).get("visitorteam", 0)),
            "possession_h": float(stats.get("possession", {}).get("localteam", 50)),
        }
    except (ValueError, TypeError, AttributeError):
        return None
