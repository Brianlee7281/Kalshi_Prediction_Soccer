"""Run Phase 3 engine against live data or a recorded replay.

Usage:
  PYTHONPATH=. python scripts/run_phase3.py --match-id KXEPLGAME-26MAR22NEWSUN --league EPL          # live
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/KXEPLGAME-26MAR20BOUMUN        # replay
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/KXEPLGAME-26MAR20BOUMUN --speed 10  # 10x
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/KXEPLGAME-26MAR20BOUMUN --trade # paper trade
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/KXEPLGAME-26MAR20BOUMUN --trade --bankroll 5000
  PYTHONPATH=. python scripts/run_phase3.py --tick-replay data/recordings/KXEPLGAME-26MAR20BOUMUN --bankroll 5000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import numpy as np
import structlog
from dotenv import load_dotenv
load_dotenv()

from src.calibration.step_1_3_ml_prior import compute_C_time
from src.common.types import MarketProbs, MatchPnL, Phase2Result, TickPayload, TradingMode
from src.engine.kalshi_live_poller import kalshi_live_poller
from src.engine.model import LiveMatchModel
from src.engine.odds_api_listener import odds_api_listener
from src.engine.tick_loop import tick_loop
from src.execution.execution_loop import execution_loop
from src.execution.mock_db import MockDBPool
from src.prematch.phase2_pipeline import backsolve_intensities
from src.recorder.recorder import MatchRecorder

log = structlog.get_logger("run_phase3")

LEAGUE_IDS: dict[str, int] = {
    "EPL": 1204,
    "LaLiga": 1399,
    "SerieA": 1269,
    "Bundesliga": 1229,
    "Ligue1": 1221,
    "MLS": 1440,
    "Brasileirao": 1141,
    "Argentina": 1081,
}


def _load_replay_metadata(replay_dir: Path) -> dict:
    """Load metadata.json from a recording directory."""
    meta_path = replay_dir / "metadata.json"
    if not meta_path.exists():
        log.error("metadata_not_found", path=str(meta_path))
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


async def _load_params_from_db(league_id: int) -> dict | None:
    """Load active production params for a league from the DB.

    Falls back to the highest version if no row has is_active=True.
    Returns a params dict with keys matching LiveMatchModel.from_phase2_result expectations,
    or None if no params found.
    """
    try:
        conn = await asyncpg.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", 5432)),
            database=os.environ.get("DB_NAME", "soccer_trading"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
        )
    except Exception as exc:
        log.warning("db_connect_failed", error=str(exc))
        return None

    try:
        row = await conn.fetchrow(
            "SELECT * FROM production_params WHERE league_id=$1 AND is_active=TRUE LIMIT 1",
            league_id,
        )
        if row is None:
            log.warning("no_active_params_fallback", league_id=league_id)
            row = await conn.fetchrow(
                "SELECT * FROM production_params WHERE league_id=$1 ORDER BY version DESC LIMIT 1",
                league_id,
            )
        if row is None:
            log.warning("no_params_found", league_id=league_id)
            return None
    finally:
        await conn.close()

    def _parse(val: str | None) -> list:
        if val is None:
            return []
        if isinstance(val, str):
            return json.loads(val)
        return val  # already parsed

    params = {
        "Q": _parse(row["q"]),
        "b": _parse(row["b"]),
        "gamma_H": _parse(row["gamma_h"]),
        "gamma_A": _parse(row["gamma_a"]),
        "delta_H": _parse(row["delta_h"]),
        "delta_A": _parse(row["delta_a"]),
        "sigma_a": float(row["sigma_a"]) if row.get("sigma_a") is not None else 0.5,
        "sigma_omega_sq": float(row["sigma_omega_sq"]) if row["sigma_omega_sq"] is not None else 0.003,
        "eta_H": float(row["eta_h"]) if row["eta_h"] is not None else 0.0,
        "eta_A": float(row["eta_a"]) if row["eta_a"] is not None else 0.0,
        "eta_H2": float(row["eta_h2"]) if row.get("eta_h2") is not None else 0.0,
        "eta_A2": float(row["eta_a2"]) if row.get("eta_a2") is not None else 0.0,
        "delta_H_pos": _parse(row["delta_h_pos"]) if row.get("delta_h_pos") else None,
        "delta_H_neg": _parse(row["delta_h_neg"]) if row.get("delta_h_neg") else None,
        "delta_A_pos": _parse(row["delta_a_pos"]) if row.get("delta_a_pos") else None,
        "delta_A_neg": _parse(row["delta_a_neg"]) if row.get("delta_a_neg") else None,
        "alpha_1": 2.0,  # not stored in DB, use default
    }
    log.info(
        "params_loaded",
        league_id=league_id,
        version=row["version"],
        brier_score=float(row["brier_score"]),
        match_count=row["match_count"],
    )
    return params


def _tickers_list_to_dict(event_ticker: str, tickers: list[str]) -> dict[str, str]:
    """Convert a flat list of market tickers to the {market_type: ticker} dict.

    Derives home/away abbreviations from the event ticker suffix.
    e.g. event_ticker="KXEPLGAME-26MAR20BOUMUN", tickers=[...-BOU, ...-MUN, ...-TIE]
    → {"home_win": "...-BOU", "away_win": "...-MUN", "draw": "...-TIE"}
    """
    # Extract team abbreviations from event ticker.
    # Format: "KXEPLGAME-26MAR20BOUMUN" → last segment "26MAR20BOUMUN"
    # Team abbrs are the last 6 chars: home=[-6:-3] "BOU", away=[-3:] "MUN"
    parts = event_ticker.split("-")
    match_abbr = parts[-1] if parts else ""
    home_abbr = match_abbr[-6:-3].upper()
    away_abbr = match_abbr[-3:].upper()

    result: dict[str, str] = {}
    for t in tickers:
        suffix = t.split("-")[-1].upper()
        if suffix == "TIE":
            result["draw"] = t
        elif suffix == home_abbr:
            result["home_win"] = t
        elif suffix == away_abbr:
            result["away_win"] = t
    return result


def _extract_kalshi_prematch_odds(
    replay_dir: Path,
    kalshi_tickers: dict[str, str],
) -> MarketProbs | None:
    """Extract pre-match implied probabilities from Kalshi orderbook snapshots.

    Reads the first orderbook_snapshot for each ticker in kalshi.jsonl,
    computes mid-prices, and normalizes to remove vig.

    Returns MarketProbs with home_win/draw/away_win, or None if data missing.
    """
    ob_path = replay_dir / "kalshi_ob.jsonl"
    if not ob_path.exists():
        log.warning("kalshi_ob_file_not_found", path=str(replay_dir))
        return None

    # Reverse map: ticker string → market type
    ticker_to_market = {v: k for k, v in kalshi_tickers.items()}
    mids: dict[str, float] = {}

    with open(ob_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "orderbook_snapshot":
                continue
            msg = record.get("msg", {})
            ticker = msg.get("market_ticker", "")
            market_type = ticker_to_market.get(ticker)
            if market_type is None or market_type in mids:
                continue

            yes_bids = msg.get("yes_dollars_fp", [])
            no_bids = msg.get("no_dollars_fp", [])

            best_bid = float(yes_bids[-1][0]) if yes_bids else None
            best_ask = (1.0 - float(no_bids[-1][0])) if no_bids else None

            if best_bid is not None and best_ask is not None:
                mids[market_type] = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mids[market_type] = best_bid
            elif best_ask is not None:
                mids[market_type] = best_ask

            if len(mids) == len(kalshi_tickers):
                break

    if not {"home_win", "draw", "away_win"}.issubset(mids):
        log.warning("kalshi_prematch_incomplete", found=list(mids.keys()))
        return None

    # Normalize to remove vig
    total = mids["home_win"] + mids["draw"] + mids["away_win"]
    implied = MarketProbs(
        home_win=mids["home_win"] / total,
        draw=mids["draw"] / total,
        away_win=mids["away_win"] / total,
    )
    log.info(
        "kalshi_prematch_odds",
        home_win=round(implied.home_win, 4),
        draw=round(implied.draw, 4),
        away_win=round(implied.away_win, 4),
    )
    return implied


async def _fetch_live_prematch_odds(
    kalshi_rest: "KalshiClient",
    kalshi_tickers: dict[str, str],
) -> tuple[MarketProbs | None, str, str]:
    """Fetch pre-match mid-prices from Kalshi REST API for live recording.

    Returns (implied MarketProbs or None, home_team, away_team).
    """
    mids: dict[str, float] = {}
    home_team = "HomeTeam"
    away_team = "AwayTeam"

    for market_type, ticker in kalshi_tickers.items():
        try:
            market = await kalshi_rest.get_market(ticker)
        except Exception as exc:
            log.warning("prematch_market_fetch_failed", ticker=ticker, error=str(exc))
            continue

        yes_bid = market.get("yes_bid_dollars")
        yes_ask = market.get("yes_ask_dollars")
        if yes_bid and yes_ask:
            mids[market_type] = (float(yes_bid) + float(yes_ask)) / 2.0
        elif yes_bid:
            mids[market_type] = float(yes_bid)
        elif yes_ask:
            mids[market_type] = float(yes_ask)

        # Extract team names from the title (e.g. "Tottenham vs Nottingham Winner?")
        title = market.get("title", "")
        if " vs " in title and home_team == "HomeTeam":
            parts = title.split(" vs ", 1)
            home_team = parts[0].strip()
            away_part = parts[1].strip()
            # Remove trailing " Winner?" or similar
            for suffix in (" Winner?", " Game"):
                if away_part.endswith(suffix):
                    away_part = away_part[: -len(suffix)].strip()
            away_team = away_part

    if not {"home_win", "draw", "away_win"}.issubset(mids):
        log.warning("live_prematch_incomplete", found=list(mids.keys()))
        return None, home_team, away_team

    total = mids["home_win"] + mids["draw"] + mids["away_win"]
    implied = MarketProbs(
        home_win=mids["home_win"] / total,
        draw=mids["draw"] / total,
        away_win=mids["away_win"] / total,
    )
    log.info(
        "live_prematch_odds",
        home_win=round(implied.home_win, 4),
        draw=round(implied.draw, 4),
        away_win=round(implied.away_win, 4),
    )
    return implied, home_team, away_team


_FALLBACK_PARAMS = {
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
    "sigma_a": 0.5,
    "sigma_omega_sq": 0.003,
    "eta_H": 0.0,
    "eta_A": 0.0,
    "eta_H2": 0.0,
    "eta_A2": 0.0,
    "alpha_1": 2.0,
}


def _make_mock_model(
    match_id: str,
    kalshi_tickers: dict[str, str] | None = None,
    params: dict | None = None,
    league_id: int = 1,
    *,
    a_H: float = 0.2,
    a_A: float = 0.1,
    mu_H: float = 1.4,
    mu_A: float = 1.1,
    C_time: float = 1.0,
    home_team: str = "HomeTeam",
    away_team: str = "AwayTeam",
    prediction_method: str = "league_mle",
    ekf_P0: float = 0.25,
    market_implied: MarketProbs | None = None,
) -> LiveMatchModel:
    """Create a LiveMatchModel for replay or live testing."""
    result = Phase2Result(
        match_id=match_id,
        league_id=league_id,
        a_H=a_H,
        a_A=a_A,
        mu_H=mu_H,
        mu_A=mu_A,
        C_time=C_time,
        verdict="GO",
        skip_reason=None,
        param_version=1,
        home_team=home_team,
        away_team=away_team,
        kickoff_utc="2026-01-01T00:00:00Z",
        kalshi_tickers=kalshi_tickers or {},
        market_implied=market_implied,
        prediction_method=prediction_method,
        ekf_P0=ekf_P0,
    )
    return LiveMatchModel.from_phase2_result(result, params or _FALLBACK_PARAMS)


async def run_live(match_id: str, league: str) -> None:
    """Run Phase 3 against live data sources.

    Connects to Kalshi live data (HTTP polling) and Kalshi WS (orderbook).
    Records all data streams to data/recordings/{match_id}/ for later
    tick-replay backtesting.
    """
    from src.clients.kalshi import KalshiClient
    from src.clients.kalshi_live_data import KalshiLiveDataClient
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.kalshi_ob_sync import kalshi_ob_sync

    league_id = LEAGUE_IDS.get(league, 1)
    db_params = await _load_params_from_db(league_id)
    if db_params is None:
        log.warning("using_fallback_params", league=league, league_id=league_id)

    # Resolve Kalshi tickers from the event ticker via REST API
    api_key = os.environ.get("KALSHI_API_KEY", "")
    private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    kalshi_rest = KalshiClient(api_key=api_key, private_key_path=private_key_path)

    # Series ticker = event ticker prefix (e.g. "KXEPLGAME" from "KXEPLGAME-26MAR20BOUMUN")
    series_ticker = match_id.split("-")[0] if "-" in match_id else match_id
    markets = await kalshi_rest.get_markets(series_ticker=series_ticker)
    tickers_list = [m["ticker"] for m in markets if m.get("event_ticker") == match_id]
    kalshi_tickers = _tickers_list_to_dict(match_id, tickers_list) if tickers_list else {}

    live_client = KalshiLiveDataClient()

    if not kalshi_tickers:
        log.error("no_kalshi_tickers", match_id=match_id)
        await live_client.close()
        return

    # Backsolve intensities from live pre-match orderbook
    backsolve_kwargs: dict = {}
    use_params = db_params or _FALLBACK_PARAMS
    implied, home_team, away_team = await _fetch_live_prematch_odds(
        kalshi_rest, kalshi_tickers,
    )
    if implied is not None:
        b = np.array(use_params["b"])
        Q = np.array(use_params["Q"])
        alpha_1 = use_params.get("alpha_1", 0.0)
        basis_bounds: np.ndarray | None = None
        if len(b) == 8:
            basis_bounds = np.array(
                [0.0, 15.0, 30.0,
                 45.0 + alpha_1, 60.0 + alpha_1, 75.0 + alpha_1,
                 85.0 + alpha_1, 90.0 + alpha_1, 93.0],
                dtype=np.float64,
            )
        a_H, a_A = backsolve_intensities(implied, b, Q, basis_bounds)
        C_time = compute_C_time(b, basis_bounds)
        mu_H = float(np.exp(a_H) * C_time)
        mu_A = float(np.exp(a_A) * C_time)
        backsolve_kwargs = dict(
            a_H=a_H, a_A=a_A, mu_H=mu_H, mu_A=mu_A, C_time=C_time,
            prediction_method="backsolve_kalshi", ekf_P0=0.15,
            market_implied=implied,
        )
        log.info(
            "live_backsolve",
            a_H=round(a_H, 4), a_A=round(a_A, 4),
            mu_H=round(mu_H, 4), mu_A=round(mu_A, 4),
        )

    await kalshi_rest.close()

    model = _make_mock_model(
        match_id,
        kalshi_tickers=kalshi_tickers,
        params=db_params,
        league_id=league_id,
        home_team=home_team,
        away_team=away_team,
        **backsolve_kwargs,
    )
    model.kalshi_event_ticker = match_id

    recorder = MatchRecorder(match_id)
    recorder.set_match_info(
        event_ticker=match_id,
        league=league,
        home_team=home_team,
        away_team=away_team,
        kalshi_tickers=list(kalshi_tickers.values()),
    )
    model.recorder = recorder  # type: ignore[attr-defined]

    # Kalshi WS client for orderbook sync (populates model.p_kalshi)
    ws_client = KalshiWSClient(
        api_key=os.environ.get("KALSHI_API_KEY", ""),
        private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH", ""),
    )

    log.info(
        "phase3_live_start",
        match_id=match_id,
        league=league,
        home=home_team,
        away=away_team,
        tickers=kalshi_tickers,
    )

    try:
        tick_task = asyncio.create_task(tick_loop(model))
        poller_task = asyncio.create_task(
            kalshi_live_poller(model, client=live_client)
        )
        ob_task = asyncio.create_task(kalshi_ob_sync(model, ws_client=ws_client))
        odds_task = asyncio.create_task(odds_api_listener(model))

        await tick_task

        for task in (poller_task, ob_task, odds_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(poller_task, ob_task, odds_task, return_exceptions=True)
    finally:
        await ws_client.disconnect()
        await live_client.close()
        recorder.finalize()
        log.info("phase3_live_done", match_id=match_id, ticks=model.tick_count)


async def run_replay(
    replay_dir: Path,
    speed: float,
    *,
    trade: bool = False,
    bankroll: float = 10_000.0,
) -> None:
    """Run Phase 3 against replayed recorded data, optionally with Phase 4 paper trading."""
    from src.clients.kalshi_live_data import KalshiLiveDataClient
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.kalshi_ob_sync import kalshi_ob_sync
    from src.recorder.replay_server import ReplayServer

    metadata = _load_replay_metadata(replay_dir)
    match_id = metadata.get("event_ticker", metadata.get("match_id", replay_dir.name))

    # Resolve league_id and load real params from DB
    league_str = metadata.get("league", "")
    league_id = LEAGUE_IDS.get(league_str, 1)
    db_params = await _load_params_from_db(league_id)
    if db_params is None:
        log.warning("using_fallback_params", league=league_str, league_id=league_id)

    raw_tickers = metadata.get("kalshi_tickers", [])
    kalshi_tickers = _tickers_list_to_dict(match_id, raw_tickers) if raw_tickers else {}

    # Backsolve a_H, a_A from Kalshi pre-match orderbook
    backsolve_kwargs: dict = {}
    implied = _extract_kalshi_prematch_odds(replay_dir, kalshi_tickers)
    use_params = db_params or _FALLBACK_PARAMS
    if implied is not None:
        b = np.array(use_params["b"])
        Q = np.array(use_params["Q"])
        alpha_1 = use_params.get("alpha_1", 0.0)
        basis_bounds: np.ndarray | None = None
        if len(b) == 8:
            basis_bounds = np.array(
                [0.0, 15.0, 30.0,
                 45.0 + alpha_1, 60.0 + alpha_1, 75.0 + alpha_1,
                 85.0 + alpha_1, 90.0 + alpha_1, 93.0],
                dtype=np.float64,
            )
        a_H, a_A = backsolve_intensities(implied, b, Q, basis_bounds)
        C_time = compute_C_time(b, basis_bounds)
        mu_H = float(np.exp(a_H) * C_time)
        mu_A = float(np.exp(a_A) * C_time)
        log.info(
            "backsolve_from_kalshi",
            a_H=round(a_H, 4),
            a_A=round(a_A, 4),
            mu_H=round(mu_H, 4),
            mu_A=round(mu_A, 4),
            C_time=round(C_time, 4),
        )
        backsolve_kwargs = dict(
            a_H=a_H, a_A=a_A, mu_H=mu_H, mu_A=mu_A, C_time=C_time,
            prediction_method="backsolve_kalshi", ekf_P0=0.15,
            market_implied=implied,
        )

    home_team = metadata.get("home_team", "HomeTeam")
    away_team = metadata.get("away_team", "AwayTeam")

    model = _make_mock_model(
        match_id,
        kalshi_tickers=kalshi_tickers,
        params=db_params,
        league_id=league_id,
        home_team=home_team,
        away_team=away_team,
        **backsolve_kwargs,
    )
    # Start in FIRST_HALF for replay (skip waiting for kickoff)
    model.engine_phase = "FIRST_HALF"

    # Attach recorder to save replay results
    output_dir = Path("data/replay_results")
    recorder = MatchRecorder(match_id, base_dir=output_dir)
    model.recorder = recorder  # type: ignore[attr-defined]

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

    # Phase 4 paper trading setup
    db_pool: MockDBPool | None = None
    if trade:
        db_pool = MockDBPool(initial_bankroll=bankroll)
        log.info("phase4_paper_trading_enabled", bankroll=bankroll)

    try:
        phase4_queue: asyncio.Queue = asyncio.Queue()

        # Create tasks so we can cancel stragglers after match ends
        tick_task = asyncio.create_task(
            tick_loop(model, phase4_queue=phase4_queue, tick_interval=0.0)
        )
        poller_task = asyncio.create_task(
            kalshi_live_poller(model, client=live_client, poll_interval=1.0 / speed, replay_mode=True)
        )
        ob_task = asyncio.create_task(kalshi_ob_sync(model, ws_client=ws_client))
        odds_task = asyncio.create_task(odds_api_listener(model))

        exec_task: asyncio.Task | None = None
        if trade and db_pool is not None:
            exec_task = asyncio.create_task(
                execution_loop(phase4_queue, model, db_pool, TradingMode.PAPER)
            )

        # Wait for tick_loop to finish (signals match end)
        await tick_task

        # If trading, send FINISHED sentinel and wait for execution_loop settlement
        match_pnl_result: MatchPnL | None = None
        if exec_task is not None:
            # tick_loop exits without putting a FINISHED payload on the queue,
            # so we inject one to unblock execution_loop's settlement path.
            await phase4_queue.put(
                _make_finished_sentinel(model.match_id, model.t, tuple(model.score))
            )
            match_pnl_result = await exec_task

        # Cancel remaining WS coroutines (they may block on recv)
        for task in (poller_task, ob_task, odds_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(poller_task, ob_task, odds_task, return_exceptions=True)

        # Display Phase 4 results
        if match_pnl_result is not None and db_pool is not None:
            _save_trade_results(output_dir / match_id, match_pnl_result, db_pool)
            _print_trade_summary(match_pnl_result, db_pool, bankroll, home_team, away_team)
    finally:
        await ws_client.disconnect()
        recorder.finalize()
        await live_client.close()
        await server.stop()
        log.info(
            "phase3_replay_done",
            match_id=match_id,
            ticks=model.tick_count,
            final_score=model.score,
            output_dir=str(output_dir / match_id),
        )


def _make_finished_sentinel(match_id: str, t: float, score: tuple[int, int]) -> TickPayload:
    """Build a FINISHED TickPayload sentinel for execution_loop settlement."""
    _zero = MarketProbs(home_win=0.0, draw=0.0, away_win=0.0)
    return TickPayload(
        match_id=match_id, t=t, engine_phase="FINISHED",
        P_model=_zero, sigma_MC=_zero,
        score=score, X=0, delta_S=0,
        mu_H=0.0, mu_A=0.0, a_H_current=0.0, a_A_current=0.0,
        ekf_P_H=0.0, ekf_P_A=0.0, hmm_state=0, dom_index=0.0,
        surprise_score=0.0, order_allowed=False, cooldown=False,
        ob_freeze=False, event_state="none",
    )


async def run_tick_replay(
    recording_dir: Path,
    *,
    bankroll: float = 10_000.0,
    speed: float = 0.0,
) -> None:
    """Run Phase 4 paper trading by replaying recorded ticks directly.

    Reads ticks.jsonl (TickPayload + p_kalshi) and feeds execution_loop.
    No Phase 3 recomputation — uses exact recorded P_model, EKF state, etc.

    Args:
        recording_dir: Path containing ticks.jsonl and metadata.json.
        bankroll: Starting paper bankroll.
        speed: Replay pacing. 0 = instant (no sleep), 1 = real-time,
               10 = 10x speed. Based on _ts spacing between ticks.
    """
    metadata = _load_replay_metadata(recording_dir)
    match_id = metadata.get("event_ticker", metadata.get("match_id", recording_dir.name))

    ticks_path = recording_dir / "ticks.jsonl"
    if not ticks_path.exists():
        log.error("ticks_file_not_found", path=str(ticks_path))
        sys.exit(1)

    # Resolve kalshi_tickers — check metadata, then derive from first tick's p_kalshi
    raw_tickers = metadata.get("kalshi_tickers", [])
    if isinstance(raw_tickers, dict):
        kalshi_tickers = raw_tickers
    elif raw_tickers:
        kalshi_tickers = _tickers_list_to_dict(match_id, raw_tickers)
    else:
        # Derive tickers from the first tick's p_kalshi keys + event ticker pattern
        kalshi_tickers = {}
        with open(ticks_path, encoding="utf-8") as f:
            first_tick = json.loads(f.readline())
            p_kalshi_keys = list(first_tick.get("p_kalshi", {}).keys())
            if p_kalshi_keys:
                # Extract team abbrs: KXEPLGAME-26MAR20BOUMUN → BOU, MUN
                parts = match_id.split("-")
                match_abbr = parts[-1] if parts else ""
                home_abbr = match_abbr[-6:-3].upper()
                away_abbr = match_abbr[-3:].upper()
                abbr_map = {
                    "home_win": home_abbr,
                    "away_win": away_abbr,
                    "draw": "TIE",
                }
                for market_type in p_kalshi_keys:
                    suffix = abbr_map.get(market_type)
                    if suffix:
                        kalshi_tickers[market_type] = f"{match_id}-{suffix}"

    home_team = metadata.get("home_team", "HomeTeam")
    away_team = metadata.get("away_team", "AwayTeam")

    log.info(
        "tick_replay_start",
        match_id=match_id,
        ticks_path=str(ticks_path),
        bankroll=bankroll,
        speed=speed,
        tickers=kalshi_tickers,
    )

    # Minimal model — execution_loop only reads p_kalshi, kalshi_tickers, match_id
    model = _make_mock_model(match_id, kalshi_tickers=kalshi_tickers)
    db_pool = MockDBPool(initial_bankroll=bankroll)
    phase4_queue: asyncio.Queue = asyncio.Queue()

    exec_task = asyncio.create_task(
        execution_loop(phase4_queue, model, db_pool, TradingMode.PAPER)
    )

    last_score: tuple[int, int] = (0, 0)
    last_t: float = 0.0
    prev_ts: float | None = None
    tick_count = 0

    with open(ticks_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Extract and set p_kalshi on model before queueing the tick
            p_kalshi = record.pop("p_kalshi", {})
            model.p_kalshi = p_kalshi

            # Pacing: sleep based on _ts spacing between ticks
            ts = record.pop("_ts", None)
            if speed > 0 and ts is not None and prev_ts is not None:
                delay = (ts - prev_ts) / speed
                if delay > 0:
                    await asyncio.sleep(delay)
            prev_ts = ts

            # score comes as list from JSON, TickPayload expects tuple
            if "score" in record and isinstance(record["score"], list):
                record["score"] = tuple(record["score"])

            payload = TickPayload(**record)
            last_score = payload.score
            last_t = payload.t
            tick_count += 1

            await phase4_queue.put(payload)

    # Inject FINISHED sentinel so execution_loop settles
    await phase4_queue.put(_make_finished_sentinel(match_id, last_t, last_score))
    match_pnl = await exec_task

    # Save and display results
    output_dir = Path("data/replay_results")
    _save_trade_results(output_dir / match_id, match_pnl, db_pool)
    _print_trade_summary(match_pnl, db_pool, bankroll, home_team, away_team)

    log.info(
        "tick_replay_done",
        match_id=match_id,
        ticks=tick_count,
        final_score=list(last_score),
    )


def _save_trade_results(
    result_dir: Path, match_pnl: MatchPnL, db_pool: MockDBPool
) -> None:
    """Save Phase 4 trade results to the replay output directory."""
    result_dir.mkdir(parents=True, exist_ok=True)

    # PnL summary
    pnl_path = result_dir / "pnl_summary.json"
    with open(pnl_path, "w", encoding="utf-8") as f:
        json.dump(match_pnl.model_dump(), f, indent=2, default=str)

    # All trades from MockDBPool (entry + exit details)
    trades_path = result_dir / "trades.jsonl"
    with open(trades_path, "w", encoding="utf-8") as f:
        for pos in db_pool.positions:
            f.write(json.dumps(pos, default=str) + "\n")

    # Bankroll trajectory
    bankroll_path = result_dir / "bankroll.jsonl"
    with open(bankroll_path, "w", encoding="utf-8") as f:
        for snap in db_pool.snapshots:
            f.write(json.dumps(snap, default=str) + "\n")

    log.info(
        "trade_results_saved",
        pnl_summary=str(pnl_path),
        trades=str(trades_path),
        n_trades=len(db_pool.positions),
    )


def _print_trade_summary(
    match_pnl: MatchPnL,
    db_pool: MockDBPool,
    initial_bankroll: float,
    home_team: str,
    away_team: str,
) -> None:
    """Print a human-readable trade summary to stdout."""
    # Build summary as a single string to avoid colorama recursion on Windows
    lines: list[str] = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  Phase 4 Paper Trading Summary")
    lines.append(f"  {home_team} vs {away_team}")
    lines.append(f"{'='*65}")

    lines.append(f"\n  Bankroll: {initial_bankroll:,.2f} -> {db_pool.bankroll:,.2f}")
    lines.append(f"  Total PnL: {match_pnl.total_pnl:+,.2f}")

    # Count trades from DB (includes pre-settlement exits)
    closed = [p for p in db_pool.positions if p.get("status") == "CLOSED"]
    total_realized = sum(p.get("realized_pnl", 0.0) or 0.0 for p in closed)
    wins = sum(1 for p in closed if (p.get("realized_pnl") or 0) > 0)
    losses = sum(1 for p in closed if (p.get("realized_pnl") or 0) < 0)

    lines.append(f"  Trades: {len(closed)}  |  Wins: {wins}  |  Losses: {losses}")
    lines.append(f"  Realized PnL (all exits): {total_realized:+,.2f}")

    if closed:
        roi = (total_realized / initial_bankroll) * 100
        lines.append(f"  ROI: {roi:+.2f}%")

    # Show individual trades
    if db_pool.positions:
        lines.append(f"\n  {'Market':<12} {'Dir':<8} {'Qty':>4} {'Entry':>7} {'Exit':>7} {'PnL':>8} {'Reason'}")
        lines.append(f"  {'-'*65}")
        for pos in db_pool.positions:
            market = pos.get("ticker", "?").split("-")[-1]
            direction = pos.get("direction", "?")
            qty = pos.get("quantity", 0)
            entry = pos.get("entry_price", 0.0)
            exit_p = pos.get("exit_price")
            pnl = pos.get("realized_pnl")
            reason = pos.get("exit_reason", "?")
            exit_str = f"{exit_p:.4f}" if exit_p is not None else "  open"
            pnl_str = f"{pnl:+.2f}" if pnl is not None else "    -"
            lines.append(f"  {market:<12} {direction:<8} {qty:>4} {entry:.4f} {exit_str} {pnl_str} {reason}")

    lines.append("")

    # Write all at once to avoid colorama recursion issues on Windows
    import sys
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 engine")
    parser.add_argument("--match-id", type=str, help="Live match ID")
    parser.add_argument("--league", type=str, help="League code (e.g. EPL)")
    parser.add_argument("--replay", type=str, help="Path to recording directory (full Phase 3 re-sim)")
    parser.add_argument("--tick-replay", type=str,
                        help="Path to recording directory with ticks.jsonl (direct tick replay, no Phase 3)")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (0 = instant)")
    parser.add_argument("--trade", action="store_true", help="Enable Phase 4 paper trading")
    parser.add_argument("--bankroll", type=float, default=10_000.0,
                        help="Starting bankroll for paper trading (default: $10,000)")
    args = parser.parse_args()

    if args.tick_replay:
        asyncio.run(run_tick_replay(
            Path(args.tick_replay),
            bankroll=args.bankroll,
            speed=args.speed if args.speed != 1.0 else 0.0,
        ))
    elif args.replay:
        asyncio.run(run_replay(
            Path(args.replay), args.speed,
            trade=args.trade, bankroll=args.bankroll,
        ))
    elif args.match_id and args.league:
        asyncio.run(run_live(args.match_id, args.league))
    else:
        parser.error("Provide --tick-replay, --replay, or both --match-id and --league")


if __name__ == "__main__":
    main()
