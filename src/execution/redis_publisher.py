"""Phase 4 Redis publishing helpers.

Publishes position updates and signals to Redis channels.
All functions catch exceptions to prevent Redis failures from crashing execution.
"""

from __future__ import annotations

import time

import structlog

from src.common.types import (
    FillResult,
    Position,
    PositionUpdateMessage,
    Signal,
    SignalMessage,
)

log = structlog.get_logger("redis_publisher")


async def publish_position_update(
    redis_client: object, position: Position, update_type: str
) -> None:
    """Publish a position update to the 'position_update' channel."""
    try:
        msg = PositionUpdateMessage(
            type=update_type,
            match_id=position.match_id,
            ticker=position.ticker,
            direction=position.direction,
            quantity=position.quantity,
            price=position.entry_price,
        )
        publish = getattr(redis_client, "publish", None)
        if publish is not None:
            await publish("position_update", msg.model_dump_json())
    except Exception as exc:
        log.warning("redis_position_update_error", error=str(exc))


async def publish_signal(
    redis_client: object, signal: Signal, fill: FillResult
) -> None:
    """Publish a signal message to 'signal:{match_id}' channel."""
    try:
        msg = SignalMessage(
            type="signal",
            match_id=signal.match_id,
            ticker=signal.ticker,
            direction=signal.direction,
            EV=signal.EV,
            P_kalshi=signal.P_kalshi,
            kelly_fraction=signal.kelly_fraction,
            fill_qty=fill.quantity,
            fill_price=fill.price,
            timestamp=time.time(),
        )
        publish = getattr(redis_client, "publish", None)
        if publish is not None:
            await publish(f"signal:{signal.match_id}", msg.model_dump_json())
    except Exception as exc:
        log.warning("redis_signal_publish_error", error=str(exc))
