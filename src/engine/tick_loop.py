"""tick_loop — main 1-second pricing loop for Phase 3.

Ties together MC pricing, OddsConsensus, and event state into a
TickPayload every second. Uses absolute time scheduling (Pattern 3)
and the signal hierarchy (Pattern 1) for P_reference selection.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import (
    MarketProbs,
    OddsConsensusResult,
    TickMessage,
    TickPayload,
)
from src.engine.mc_pricing import compute_mc_prices

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.tick_loop")

# Tolerance for LOW confidence cross-check (Pattern 1)
_LOW_CONFIDENCE_THRESHOLD = 0.10


async def tick_loop(
    model: LiveMatchModel,
    phase4_queue: asyncio.Queue | None = None,
    redis_client: object | None = None,
) -> None:
    """Main tick loop. Every 1 second:

    1. Update model.t from wall clock
    2. Compute MC prices -> P_model, sigma_MC
    3. Get OddsConsensus -> P_consensus
    4. Select P_reference (Pattern 1)
    5. Build TickPayload
    6. Send to Phase 4 queue
    7. Publish to Redis
    8. Record tick if recorder attached

    Tick scheduling: absolute time, not sleep(1) (Pattern 3).
    """
    start_time = time.monotonic()

    while model.engine_phase != "FINISHED":
        model.tick_count += 1

        # Cooldown management
        if model.cooldown and model.tick_count >= model.cooldown_until_tick:
            model.cooldown = False
            model.event_state = "IDLE"

        # Phase-dependent behavior
        if model.engine_phase == "WAITING_FOR_KICKOFF":
            await _sleep_until_next_tick(start_time, model.tick_count)
            continue

        if model.engine_phase == "HALFTIME":
            await _sleep_until_next_tick(start_time, model.tick_count)
            continue

        # FIRST_HALF or SECOND_HALF: full pricing pipeline
        model.update_time()

        # MC pricing (runs in thread executor)
        P_model, sigma_MC = await compute_mc_prices(model)

        # OddsConsensus
        consensus_result: OddsConsensusResult | None = None
        if model.odds_consensus is not None:
            consensus_result = model.odds_consensus.compute_reference()

        # Select P_reference (Pattern 1)
        P_reference, reference_source = select_P_reference(consensus_result, P_model)

        # Build TickPayload
        payload = TickPayload(
            match_id=model.match_id,
            t=model.t,
            engine_phase=model.engine_phase,
            odds_consensus=consensus_result,
            P_model=P_model,
            sigma_MC=sigma_MC,
            P_reference=P_reference,
            reference_source=reference_source,
            score=model.score,
            X=model.current_state_X,
            delta_S=model.delta_S,
            mu_H=model.mu_H,
            mu_A=model.mu_A,
            a_H_current=model.a_H,
            a_A_current=model.a_A,
            last_goal_type=model.last_goal_type,
            order_allowed=model.order_allowed,
            cooldown=model.cooldown,
            ob_freeze=model.ob_freeze,
            event_state=model.event_state,
        )

        # Send to Phase 4 queue
        if phase4_queue is not None:
            await phase4_queue.put(payload)

        # Publish to Redis
        if redis_client is not None:
            await _publish_tick_to_redis(model, payload, redis_client)

        # Record tick
        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_tick(payload)

        logger.debug(
            "tick",
            tick=model.tick_count,
            t=round(model.t, 2),
            ref_source=reference_source,
            hw=round(P_reference.home_win, 4),
            order_allowed=model.order_allowed,
        )

        await _sleep_until_next_tick(start_time, model.tick_count)

    # Send final FINISHED payload
    logger.info("tick_loop_finished", match_id=model.match_id, ticks=model.tick_count)


def select_P_reference(
    odds_consensus: OddsConsensusResult | None,
    P_model: MarketProbs,
) -> tuple[MarketProbs, str]:
    """Select P_reference from consensus or model.

    Pattern 1 — Signal Hierarchy:
    - HIGH confidence: use consensus
    - LOW confidence: use consensus if agrees with model (±10%), else model
    - NONE / None: use model
    """
    if odds_consensus is None or odds_consensus.confidence == "NONE":
        return P_model, "model"

    if odds_consensus.confidence == "HIGH":
        return odds_consensus.P_consensus, "consensus"

    # LOW confidence: cross-check with model
    if odds_consensus.confidence == "LOW":
        diff = abs(odds_consensus.P_consensus.home_win - P_model.home_win)
        if diff <= _LOW_CONFIDENCE_THRESHOLD:
            return odds_consensus.P_consensus, "consensus"
        return P_model, "model"

    return P_model, "model"


async def _publish_tick_to_redis(
    model: LiveMatchModel,
    payload: TickPayload,
    redis_client: object,
) -> None:
    """Publish TickMessage to Redis channel 'tick:{match_id}'."""
    consensus_confidence = "NONE"
    if payload.odds_consensus is not None:
        consensus_confidence = payload.odds_consensus.confidence

    msg = TickMessage(
        match_id=payload.match_id,
        t=payload.t,
        engine_phase=payload.engine_phase,
        P_reference=payload.P_reference,
        reference_source=payload.reference_source,
        P_model=payload.P_model,
        sigma_MC=payload.sigma_MC,
        consensus_confidence=consensus_confidence,
        order_allowed=payload.order_allowed,
        cooldown=payload.cooldown,
        ob_freeze=payload.ob_freeze,
        event_state=payload.event_state,
        mu_H=payload.mu_H,
        mu_A=payload.mu_A,
        score=payload.score,
    )

    channel = f"tick:{model.match_id}"
    try:
        publish = getattr(redis_client, "publish", None)
        if publish is not None:
            await publish(channel, msg.model_dump_json())
    except Exception as exc:
        logger.warning("redis_publish_error", channel=channel, error=str(exc))


async def _write_tick_snapshot(
    model: LiveMatchModel,
    payload: TickPayload,
) -> None:
    """INSERT into tick_snapshots table.

    Placeholder — will be implemented when DB layer is available.
    """
    pass


async def _sleep_until_next_tick(
    start_time: float,
    tick_count: int,
    interval: float = 1.0,
) -> None:
    """Sleep until next absolute tick time. Skip if already past."""
    next_tick_time = start_time + tick_count * interval
    sleep_duration = next_tick_time - time.monotonic()
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)
