"""Run Phase 3 engine against live data or a recorded replay.

Usage:
  PYTHONPATH=. python scripts/run_phase3.py --match-id 12345 --league EPL          # live
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/match_12345   # replay
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/match_12345 --speed 10  # 10x
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

from src.common.types import Phase2Result
from src.engine.goalserve_poller import goalserve_poller
from src.engine.model import LiveMatchModel
from src.engine.odds_api_listener import odds_api_listener
from src.engine.odds_consensus import OddsConsensus
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
        "b": [0.1, 0.15, 0.12, 0.08, 0.10, -0.05],
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
    model.odds_consensus = OddsConsensus()
    recorder = MatchRecorder(match_id)
    model.recorder = recorder  # type: ignore[attr-defined]

    log.info("phase3_live_start", match_id=match_id, league=league)

    try:
        await asyncio.gather(
            tick_loop(model),
            odds_api_listener(model),
            goalserve_poller(model),
        )
    finally:
        recorder.finalize()
        log.info("phase3_live_done", match_id=match_id, ticks=model.tick_count)


async def run_replay(replay_dir: Path, speed: float) -> None:
    """Run Phase 3 against replayed recorded data."""
    from src.recorder.replay_server import ReplayServer

    metadata = _load_replay_metadata(replay_dir)
    match_id = metadata.get("match_id", replay_dir.name)

    model = _make_mock_model(match_id)
    model.odds_consensus = OddsConsensus()
    # Start in FIRST_HALF for replay (skip waiting for kickoff)
    model.engine_phase = "FIRST_HALF"

    server = ReplayServer(replay_dir, speed=speed)
    await server.start()

    log.info(
        "phase3_replay_start",
        match_id=match_id,
        speed=speed,
        goalserve_port=server.goalserve_port,
        odds_ws_port=server.odds_ws_port,
    )

    try:
        phase4_queue: asyncio.Queue = asyncio.Queue()
        await tick_loop(model, phase4_queue=phase4_queue)
    finally:
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
