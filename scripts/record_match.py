"""Record a live match to JSONL without trading.

Wires kalshi_live_poller + kalshi_ob_sync + tick_loop with a MatchRecorder
attached to the model. No Phase 4 execution is started.

Usage:
  PYTHONPATH=. python scripts/record_match.py \\
    --match-id 4346261 \\
    --event-ticker KXEPLGAME-25MAR17-MANCI \\
    --home "Manchester City" \\
    --away "Real Madrid" \\
    --league-id 1204 \\
    --a-h 0.35 \\
    --a-a 0.28

Required environment variables:
  KALSHI_API_KEY
  KALSHI_PRIVATE_KEY_PATH
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.clients.kalshi_ws import KalshiWSClient
from src.common.logging import get_logger
from src.engine.kalshi_live_poller import kalshi_live_poller
from src.engine.kalshi_ob_sync import kalshi_ob_sync
from src.engine.model import LiveMatchModel, _precompute_grids
from src.engine.tick_loop import tick_loop
from src.recorder.recorder import MatchRecorder

log = get_logger("record_match")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a live match (no trading)."
    )
    parser.add_argument("--match-id", required=True, type=str,
                        help="Goalserve/internal match ID")
    parser.add_argument("--event-ticker", required=True, type=str,
                        help="Kalshi event-level ticker (for live_data milestone lookup)")
    parser.add_argument("--home", required=True, type=str,
                        help="Home team name")
    parser.add_argument("--away", required=True, type=str,
                        help="Away team name")
    parser.add_argument("--league-id", required=True, type=int,
                        help="Internal league ID")
    parser.add_argument("--a-h", required=True, type=float,
                        help="Home log-intensity (a_H) from Phase 2 backsolve")
    parser.add_argument("--a-a", required=True, type=float,
                        help="Away log-intensity (a_A) from Phase 2 backsolve")
    return parser.parse_args()


def _load_env() -> tuple[str, str]:
    """Read and validate required Kalshi credentials from environment."""
    api_key = os.environ.get("KALSHI_API_KEY", "")
    private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    missing = [
        name
        for name, val in (
            ("KALSHI_API_KEY", api_key),
            ("KALSHI_PRIVATE_KEY_PATH", private_key_path),
        )
        if not val
    ]
    if missing:
        log.error("missing_env_vars", vars=missing)
        sys.exit(1)
    return api_key, private_key_path


def _build_model(args: argparse.Namespace) -> LiveMatchModel:
    """Build a minimal LiveMatchModel with stub calibration params.

    Stubs are valid but uninformative — sufficient for recording and model
    validation, not for trading decisions.
    """
    Q = np.zeros((4, 4))
    b = np.zeros(8)
    gamma_H = np.ones(4)
    gamma_A = np.ones(4)
    delta_H = np.zeros(5)
    delta_A = np.zeros(5)
    basis_bounds = np.linspace(0, 90, 9)

    P_grid, P_fine_grid = _precompute_grids(Q)

    return LiveMatchModel(
        match_id=args.match_id,
        league_id=args.league_id,
        home_team=args.home,
        away_team=args.away,
        a_H=args.a_h,
        a_A=args.a_a,
        param_version=0,
        b=b,
        gamma_H=gamma_H,
        gamma_A=gamma_A,
        delta_H=delta_H,
        delta_A=delta_A,
        Q=Q,
        basis_bounds=basis_bounds,
        kalshi_tickers={},
        kalshi_event_ticker=args.event_ticker,
        P_grid=P_grid,
        P_fine_grid=P_fine_grid,
    )


async def _record(
    args: argparse.Namespace,
    api_key: str,
    private_key_path: str,
) -> None:
    """Run the three Phase 3 coroutines with a recorder attached."""
    model = _build_model(args)
    recorder = MatchRecorder(match_id=args.match_id)
    model.recorder = recorder  # type: ignore[attr-defined]

    ws_client = KalshiWSClient(api_key=api_key, private_key_path=private_key_path)

    output_dir = Path("data/recordings") / args.match_id
    print(f"Recording to: {output_dir}/")

    log.info(
        "record_match_start",
        match_id=args.match_id,
        event_ticker=args.event_ticker,
        home=args.home,
        away=args.away,
        league_id=args.league_id,
    )

    try:
        await asyncio.gather(
            kalshi_live_poller(model),
            kalshi_ob_sync(model, ws_client),
            tick_loop(model, phase4_queue=None, redis_client=None),
        )
    finally:
        recorder.finalize()
        await ws_client.disconnect()
        log.info("record_match_done", match_id=args.match_id, ticks=model.tick_count)


def main() -> None:
    args = _parse_args()
    api_key, private_key_path = _load_env()
    asyncio.run(_record(args, api_key, private_key_path))


if __name__ == "__main__":
    main()
