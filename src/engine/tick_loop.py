"""tick_loop — main 1-second pricing loop for Phase 3 (v5).

v5 pipeline (7 steps per tick):
  1. Update effective match time
  2. EKF prediction step (uncertainty grows)
  3. No-goal EKF update (weak negative evidence)
  4. Layer 2 already updated by goalserve_poller
  5. MC simulation → P_model
  6. Compute σ²_p (total probability uncertainty)
  7. Assemble TickPayload → Phase 4 queue + Redis

v5 removes OddsConsensus/P_reference/signal hierarchy.
P_model is the sole trading authority.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import MarketProbs, TickPayload
from src.engine.intensity import basis_index as _basis_index, compute_lambda as _compute_lambda
from src.engine.mc_pricing import compute_mc_prices

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.tick_loop")


async def tick_loop(
    model: LiveMatchModel,
    phase4_queue: asyncio.Queue | None = None,
    redis_client: object | None = None,
    tick_interval: float = 1.0,
) -> None:
    """Main tick loop — v5 7-step pipeline every tick_interval seconds.

    Args:
        tick_interval: Seconds between ticks. Default 1.0 for live.
                       Use 0.0 for replay (runs as fast as MC allows).
    """
    start_time = time.monotonic()

    while model.engine_phase != "FINISHED":
        model.tick_count += 1

        # Cooldown management
        if model.cooldown and model.tick_count >= model.cooldown_until_tick:
            model.cooldown = False
            model.event_state = "IDLE"

        # Skip pricing during inactive phases
        if model.engine_phase in ("WAITING_FOR_KICKOFF", "HALFTIME"):
            await _sleep_until_next_tick(start_time, model.tick_count, interval=tick_interval)
            continue

        # ── v5 7-step pipeline ──────────────────────────────

        # Step 1: Update effective match time
        # In replay mode (tick_interval=0), poller drives model.t directly
        if tick_interval > 0:
            model.update_time()

        # Step 2: EKF prediction step
        if model.ekf_tracker is not None:
            model.ekf_tracker.predict(dt=1.0 / 60.0)  # dt in minutes

        # Step 3: No-goal EKF update (weak negative evidence)
        if model.strength_updater is not None and model.ekf_tracker is not None:
            lambda_H = _compute_lambda(model, "home")
            lambda_A = _compute_lambda(model, "away")
            model.strength_updater.update_no_goal(lambda_H, lambda_A, dt=1.0 / 60.0)
            model.a_H = model.strength_updater.a_H
            model.a_A = model.strength_updater.a_A

        # Step 4: Layer 2 — HMM/DomIndex already updated by kalshi_live_poller

        # Step 5: MC simulation
        P_model, sigma_MC = await compute_mc_prices(model)

        # Step 6: σ²_p for Phase 4 (stored in sigma_MC for now)

        # Step 7: Assemble TickPayload
        payload = TickPayload(
            match_id=model.match_id,
            t=model.t,
            engine_phase=model.engine_phase,
            P_model=P_model,
            sigma_MC=sigma_MC,
            score=model.score,
            X=model.current_state_X,
            delta_S=model.delta_S,
            mu_H=model.mu_H,
            mu_A=model.mu_A,
            a_H_current=model.a_H,
            a_A_current=model.a_A,
            last_goal_type=model.last_goal_type,
            ekf_P_H=model.ekf_tracker.P_H if model.ekf_tracker else 0.0,
            ekf_P_A=model.ekf_tracker.P_A if model.ekf_tracker else 0.0,
            hmm_state=model.hmm_estimator.state if model.hmm_estimator else 0,
            dom_index=model.hmm_estimator.dom_index_value if model.hmm_estimator else 0.0,
            surprise_score=model.surprise_score,
            order_allowed=model.order_allowed,
            cooldown=model.cooldown,
            ob_freeze=model.ob_freeze,
            event_state=model.event_state,
        )

        if phase4_queue is not None:
            await phase4_queue.put(payload)

        if redis_client is not None:
            await _publish_tick_to_redis(model, payload, redis_client)

        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_tick(payload)

        logger.debug(
            "tick",
            tick=model.tick_count,
            t=round(model.t, 2),
            hw=round(P_model.home_win, 4),
            order_allowed=model.order_allowed,
        )

        await _sleep_until_next_tick(start_time, model.tick_count, interval=tick_interval)

    logger.info("tick_loop_finished", match_id=model.match_id, ticks=model.tick_count)


async def _publish_tick_to_redis(
    model: LiveMatchModel,
    payload: TickPayload,
    redis_client: object,
) -> None:
    """Publish tick to Redis."""
    channel = f"tick:{model.match_id}"
    try:
        publish = getattr(redis_client, "publish", None)
        if publish is not None:
            await publish(channel, payload.model_dump_json())
    except Exception as exc:
        logger.warning("redis_publish_error", channel=channel, error=str(exc))


async def _sleep_until_next_tick(
    start_time: float,
    tick_count: int,
    interval: float = 1.0,
) -> None:
    """Sleep until next absolute tick time. Skip if already past.

    When interval is 0 (replay mode), yields control without sleeping
    so other coroutines (poller, WS listeners) can run.
    """
    if interval <= 0:
        await asyncio.sleep(0)  # yield to event loop
        return
    next_tick_time = start_time + tick_count * interval
    sleep_duration = next_tick_time - time.monotonic()
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)
