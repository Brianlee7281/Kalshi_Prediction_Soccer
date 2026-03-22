"""Goalserve live poller — cross-validates scores against Kalshi.

Runs alongside kalshi_live_poller as a secondary, authoritative score source.
Goalserve is slower (~38s behind Kalshi prices) but more reliable for
VAR reversals and score corrections.

Kalshi is trusted for speed (initial goal detection).
Goalserve is trusted for accuracy (VAR corrections, missed events).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.engine.event_handlers import handle_goal, handle_score_correction

if TYPE_CHECKING:
    from src.clients.goalserve import GoalserveClient
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.goalserve_live_poller")

_DEFAULT_POLL_INTERVAL = 5.0  # seconds between Goalserve polls
_MISMATCH_GRACE_PERIOD = 60.0  # seconds to wait before trusting Goalserve over Kalshi


async def goalserve_live_poller(
    model: LiveMatchModel,
    goalserve_client: GoalserveClient,
    goalserve_match_id: str,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> None:
    """Coroutine: poll Goalserve live scores and cross-validate with model.

    Runs until model.engine_phase == "FINISHED".

    Args:
        model: Shared live match state (also updated by kalshi_live_poller).
        goalserve_client: Configured Goalserve API client.
        goalserve_match_id: Goalserve match ID (@id, @fix_id, or @static_id).
        poll_interval: Seconds between polls (default 5.0).
    """
    logger.info(
        "goalserve_poller_started",
        match_id=model.match_id,
        goalserve_match_id=goalserve_match_id,
        poll_interval=poll_interval,
    )

    while model.engine_phase != "FINISHED":
        try:
            live_data = await asyncio.wait_for(
                goalserve_client.get_live_scores(),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("goalserve_poll_error", error=str(exc))
            await asyncio.sleep(poll_interval)
            continue

        match = goalserve_client.find_match_in_live(goalserve_match_id, live_data)
        if match is None:
            logger.debug("goalserve_match_not_found", goalserve_id=goalserve_match_id)
            await asyncio.sleep(poll_interval)
            continue

        # Extract score from Goalserve
        gs_home = _parse_goals(match.get("localteam", {}).get("@goals", "0"))
        gs_away = _parse_goals(match.get("visitorteam", {}).get("@goals", "0"))
        gs_score = (gs_home, gs_away)

        model.goalserve_score = gs_score
        model.goalserve_last_poll_ts = time.monotonic()

        # Recording
        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            _record_goalserve(recorder, match, gs_score)

        # Cross-validate with model score
        model_score = model.score

        if gs_score == model_score:
            # Scores agree — clear any mismatch state
            if model.score_mismatch_since is not None:
                logger.info(
                    "score_mismatch_resolved",
                    score=gs_score,
                )
                model.score_mismatch_since = None
        else:
            # Scores disagree
            now = time.monotonic()

            if model.score_mismatch_since is None:
                # First detection of mismatch
                model.score_mismatch_since = now
                logger.warning(
                    "score_mismatch_detected",
                    model_score=model_score,
                    goalserve_score=gs_score,
                )
            else:
                elapsed = now - model.score_mismatch_since
                if elapsed > _MISMATCH_GRACE_PERIOD:
                    # Mismatch persisted past grace period — trust Goalserve
                    logger.critical(
                        "score_correction_from_goalserve",
                        model_score=model_score,
                        goalserve_score=gs_score,
                        mismatch_duration=round(elapsed, 1),
                    )

                    # Determine what changed
                    if gs_home + gs_away > model_score[0] + model_score[1]:
                        # Goalserve has MORE goals — Kalshi missed one
                        _apply_missing_goals(model, model_score, gs_score)
                    else:
                        # Goalserve has FEWER goals — VAR cancellation
                        handle_score_correction(model, gs_score, source="goalserve")

                    model.score_mismatch_since = None

        await asyncio.sleep(poll_interval)

    logger.info("goalserve_poller_finished", match_id=model.match_id)


def _parse_goals(goals_str: str) -> int:
    """Parse Goalserve goals string to int. Handles '?' and empty strings."""
    try:
        return int(goals_str)
    except (ValueError, TypeError):
        return 0


def _apply_missing_goals(
    model: LiveMatchModel,
    current: tuple[int, int],
    target: tuple[int, int],
) -> None:
    """Apply missing goals to reach target score."""
    home_diff = target[0] - current[0]
    away_diff = target[1] - current[1]

    for _ in range(home_diff):
        handle_goal(model, "home", int(model.t))
    for _ in range(away_diff):
        handle_goal(model, "away", int(model.t))


def _record_goalserve(recorder: object, match: dict, score: tuple[int, int]) -> None:
    """Record Goalserve poll to JSONL if recorder supports it."""
    record_fn = getattr(recorder, "record_goalserve_live_data", None)
    if record_fn is not None:
        record_fn({
            "status": match.get("@status", ""),
            "home_goals": score[0],
            "away_goals": score[1],
            "events": match.get("events", {}),
        })
