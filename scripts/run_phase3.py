"""Run Phase 3 engine against live data or a recorded replay.

Usage:
  PYTHONPATH=. python scripts/run_phase3.py --match-id 12345 --league EPL          # live
  PYTHONPATH=. python scripts/run_phase3.py --replay data/latency/KXEPLGAME-26MAR20BOUMUN        # replay
  PYTHONPATH=. python scripts/run_phase3.py --replay data/latency/KXEPLGAME-26MAR20BOUMUN --speed 10  # 10x
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv
load_dotenv()

from src.common.types import Phase2Result
from src.engine.kalshi_live_poller import kalshi_live_poller
from src.engine.model import LiveMatchModel
from src.engine.odds_api_listener import odds_api_listener
from src.engine.tick_loop import tick_loop
from src.recorder.recorder import MatchRecorder

log = structlog.get_logger("run_phase3")


def _load_replay_metadata(replay_dir: Path) -> dict:
    """Load metadata.json from a recording directory."""
    meta_path = replay_dir / "metadata.json"
    if not meta_path.exists():
        log.error("metadata_not_found", path=str(meta_path))
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def _make_mock_model(match_id: str) -> LiveMatchModel:
    """Create a minimal LiveMatchModel for replay testing."""
    # Use a mock Phase2Result with reasonable defaults
    result = Phase2Result(
        match_id=match_id,
        league_id=1,
        a_H=0.2,
        a_A=0.1,
        mu_H=1.4,
        mu_A=1.1,
        C_time=1.0,
        verdict="GO",
        skip_reason=None,
        param_version=1,
        home_team="HomeTeam",
        away_team="AwayTeam",
        kickoff_utc="2026-01-01T00:00:00Z",
        kalshi_tickers={},
        market_implied=None,
        prediction_method="league_mle",
    )
    params = {
        "Q": [
            [-0.02, 0.01, 0.01, 0.00],
            [0.00, -0.01, 0.00, 0.01],
            [0.00, 0.00, -0.01, 0.01],
            [0.00, 0.00, 0.00, 0.00],
        ],
        "b": [0.1, 0.15, 0.12, 0.08, 0.10, -0.05, 0.05, 0.0],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }
    return LiveMatchModel.from_phase2_result(result, params)


async def run_live(match_id: str, league: str) -> None:
    """Run Phase 3 against live data sources."""
    model = _make_mock_model(match_id)
    recorder = MatchRecorder(match_id)
    model.recorder = recorder  # type: ignore[attr-defined]

    log.info("phase3_live_start", match_id=match_id, league=league)

    try:
        await asyncio.gather(
            tick_loop(model),
            odds_api_listener(model),
            kalshi_live_poller(model),
        )
    finally:
        recorder.finalize()
        log.info("phase3_live_done", match_id=match_id, ticks=model.tick_count)


async def run_replay(replay_dir: Path, speed: float) -> None:
    """Run Phase 3 against replayed recorded data."""
    from src.clients.kalshi_live_data import KalshiLiveDataClient
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.kalshi_ob_sync import kalshi_ob_sync
    from src.recorder.replay_server import ReplayServer

    metadata = _load_replay_metadata(replay_dir)
    match_id = metadata.get("event_ticker", metadata.get("match_id", replay_dir.name))

    model = _make_mock_model(match_id)
    # Start in FIRST_HALF for replay (skip waiting for kickoff)
    model.engine_phase = "FIRST_HALF"

    server = ReplayServer(replay_dir, speed=speed)
    await server.start()

    # Create replay-mode clients pointing at localhost (no auth needed)
    live_client = KalshiLiveDataClient(
        base_url=f"http://127.0.0.1:{server.kalshi_live_port}",
    )
    ws_client = KalshiWSClient(
        ws_url=f"ws://127.0.0.1:{server.kalshi_ws_port}/ws",
    )

    log.info(
        "phase3_replay_start",
        match_id=match_id,
        speed=speed,
        kalshi_live_port=server.kalshi_live_port,
        odds_ws_port=server.odds_ws_port,
        kalshi_ws_port=server.kalshi_ws_port,
    )

    try:
        phase4_queue: asyncio.Queue = asyncio.Queue()
        await asyncio.gather(
            tick_loop(model, phase4_queue=phase4_queue, tick_interval=0.0),
            kalshi_live_poller(model, client=live_client, poll_interval=1.0 / speed, replay_mode=True),
            kalshi_ob_sync(model, ws_client=ws_client),
            odds_api_listener(model),
        )
    finally:
        await live_client.close()
        await server.stop()
        log.info(
            "phase3_replay_done",
            match_id=match_id,
            ticks=model.tick_count,
            final_score=model.score,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 engine")
    parser.add_argument("--match-id", type=str, help="Live match ID")
    parser.add_argument("--league", type=str, help="League code (e.g. EPL)")
    parser.add_argument("--replay", type=str, help="Path to recording directory")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    args = parser.parse_args()

    if args.replay:
        asyncio.run(run_replay(Path(args.replay), args.speed))
    elif args.match_id and args.league:
        asyncio.run(run_live(args.match_id, args.league))
    else:
        parser.error("Provide --replay or both --match-id and --league")


if __name__ == "__main__":
    main()
