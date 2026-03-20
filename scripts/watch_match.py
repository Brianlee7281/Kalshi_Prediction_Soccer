"""Minimal live match watcher — just logs Kalshi live data every second.

Usage:
  PYTHONPATH=. python scripts/watch_match.py --event-ticker KXEPLGAME-25MAR17-MANCI
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from src.clients.kalshi_live_data import KalshiLiveDataClient
from src.common.logging import get_logger

log = get_logger("watch_match")


async def _watch(event_ticker: str) -> None:
    api_key = os.environ.get("KALSHI_API_KEY", "")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
    if not api_key:
        print("Set KALSHI_API_KEY"); sys.exit(1)

    client = KalshiLiveDataClient(api_key=api_key, private_key_path=key_path)
    uuid = await client.resolve_milestone_uuid(event_ticker)
    log.info("milestone_resolved", uuid=uuid)

    try:
        while True:
            try:
                state = await client.get_live_data(uuid)
                log.info(
                    "live",
                    half=state.half,
                    minute=state.minute,
                    stoppage=state.stoppage,
                    score=f"{state.home_score}-{state.away_score}",
                    status=state.status,
                    last_play=state.last_play_desc,
                )
            except Exception as exc:
                log.warning("poll_error", error=str(exc))
            await asyncio.sleep(1.0)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-ticker", required=True)
    args = parser.parse_args()
    asyncio.run(_watch(args.event_ticker))


if __name__ == "__main__":
    main()
