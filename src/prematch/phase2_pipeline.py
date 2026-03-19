"""Phase 2 Pipeline — Pre-match initialization.

Runs at kickoff -65 minutes:
  Load params → get market odds → backsolve a_H/a_A → sanity check → Phase2Result.

Intensity prediction tier order:
  Tier 1: Odds-API (Bet365 + Betfair Exchange) → backsolve
  Tier 2: Pinnacle closing odds (football-data.co.uk) → backsolve
  Tier 3: XGBoost (last resort ML)
  Tier 4: League MLE (fallback when no odds available)
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import asyncpg
import numpy as np
import structlog
from scipy.optimize import minimize

from src.calibration.step_1_3_ml_prior import compute_C_time
from src.clients.kalshi import KalshiClient
from src.clients.kalshi_ticker_matcher import match_fixtures_to_tickers
from src.clients.odds_api import LEAGUE_SLUGS, OddsApiClient
from src.common.config import Config
from src.common.types import MarketProbs, Phase2Result

log = structlog.get_logger(__name__)

# League → Kalshi series prefix
LEAGUE_PREFIXES: dict[int, str] = {
    1204: "KXEPLGAME",
    1399: "KXLALIGAGAME",
    1269: "KXSERIEAGAME",
    1229: "KXBUNDESLIGAGAME",
    1221: "KXLIGUE1GAME",
    1440: "KXMLSGAME",
    1141: "KXBRASILEIROGAME",
    1081: "KXARGPREMDIVGAME",
}


async def run_phase2(
    match_id: str,
    league_id: int,
    home_team: str,
    away_team: str,
    kickoff_utc: datetime,
    config: Config,
) -> Phase2Result:
    """Full Phase 2 pipeline. Runs at kickoff -65 minutes.

    Steps:
    1. Load active production_params for this league from DB
    2. Collect pre-match odds (Odds-API first, then Pinnacle fallback)
    3. Backsolve a_H, a_A from market odds (tiered fallback)
    4. Compute mu_H, mu_A = exp(a_H) * C_time, exp(a_A) * C_time
    5. Sanity check: compare model P(1x2) vs market implied probs
    6. Match Kalshi tickers
    7. Build and return Phase2Result
    """
    # Step 1: Load production params
    params = await load_production_params(config, league_id)
    if params is None:
        log.error("phase2_no_params", league_id=league_id)
        return _skip_result(
            match_id, league_id, home_team, away_team, kickoff_utc,
            reason="No production_params for league",
        )

    b = np.array(params["b"])
    Q = np.array(params["Q"])

    # Construct basis_bounds so compute_C_time uses actual period widths.
    # This matches the same construction in LiveMatchModel.from_phase2_result.
    T_exp = 93.0
    alpha_1 = params.get("alpha_1", 0.0)
    basis_bounds: np.ndarray | None = None
    if len(b) == 8:
        basis_bounds = np.array(
            [0.0, 15.0, 30.0,
             45.0 + alpha_1, 60.0 + alpha_1, 75.0 + alpha_1,
             85.0 + alpha_1, 90.0 + alpha_1, T_exp],
            dtype=np.float64,
        )

    C_time = compute_C_time(b, basis_bounds)
    param_version = params["version"]

    # Step 2: Collect pre-match odds from Odds-API
    odds_api_implied: MarketProbs | None = None
    league_slug = LEAGUE_SLUGS.get(str(league_id))
    if league_slug and config.odds_api_key:
        try:
            odds_api_implied = await _fetch_market_odds(
                config, league_slug, home_team, away_team,
            )
        except Exception as exc:
            log.warning("phase2_odds_api_fetch_failed", error=str(exc))

    # Step 3: Predict a_H, a_A (tiered fallback)
    prediction_method: str
    a_H: float
    a_A: float
    market_implied: MarketProbs | None = None

    # Tier 1: Backsolve from Odds-API (Bet365 / Betfair Exchange)
    if odds_api_implied is not None:
        a_H, a_A = backsolve_intensities(odds_api_implied, b, Q, basis_bounds)
        prediction_method = "backsolve_odds_api"
        market_implied = odds_api_implied
        log.info("phase2_tier1_odds_api", a_H=a_H, a_A=a_A)
    else:
        # Tier 2: Backsolve from Pinnacle closing odds (football-data.co.uk)
        pinnacle_implied = _fetch_pinnacle_odds(league_id, home_team, away_team)
        if pinnacle_implied is not None:
            a_H, a_A = backsolve_intensities(pinnacle_implied, b, Q, basis_bounds)
            prediction_method = "backsolve_pinnacle"
            market_implied = pinnacle_implied
            log.info("phase2_tier2_pinnacle", a_H=a_H, a_A=a_A)
        else:
            # Tier 3: XGBoost from DB blob
            xgb_blob = params.get("xgb_model_blob")
            xgb_ok = False
            if xgb_blob:
                try:
                    models = pickle.loads(xgb_blob)
                    feature_mask = params.get("feature_mask")
                    n_features = len(feature_mask) if feature_mask else None
                    features = _build_features_from_market(
                        MarketProbs(home_win=0.40, draw=0.30, away_win=0.30),
                        n_features,
                    )
                    a_H = float(models["home"].predict(np.array([features]))[0])
                    a_A = float(models["away"].predict(np.array([features]))[0])
                    prediction_method = "xgboost"
                    xgb_ok = True
                    log.info("phase2_tier3_xgboost", a_H=a_H, a_A=a_A)
                except Exception as exc:
                    log.warning("phase2_xgb_predict_failed", error=str(exc))

            if not xgb_ok:
                # Tier 4: League average MLE
                a_H, a_A = _league_mle(C_time)
                prediction_method = "league_mle"
                log.info("phase2_tier4_league_mle", a_H=a_H, a_A=a_A)

    # Step 4: Compute mu
    mu_H = float(np.exp(a_H) * C_time)
    mu_A = float(np.exp(a_A) * C_time)

    # Step 5: Sanity check
    model_probs = _compute_model_probs(mu_H, mu_A)
    verdict, skip_reason = sanity_check(model_probs, market_implied)

    # Step 6: Kalshi ticker matching
    kalshi_tickers: dict[str, str] = {}
    prefix = LEAGUE_PREFIXES.get(league_id)
    if prefix and config.kalshi_api_key:
        try:
            kalshi_client = KalshiClient(
                api_key=config.kalshi_api_key,
                private_key_path=config.kalshi_private_key_path,
            )
            fixtures = [{
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": kickoff_utc,
            }]
            matched = await match_fixtures_to_tickers(fixtures, kalshi_client, prefix)
            kalshi_tickers = matched.get(match_id, {})
            await kalshi_client.close()
        except Exception as exc:
            log.warning("phase2_ticker_match_failed", error=str(exc))

    # v5: EKF initial uncertainty — tier-dependent
    ekf_P0_map = {
        "backsolve_odds_api": 0.15,   # Tier 1: Betfair/Bet365, high confidence
        "backsolve_pinnacle": 0.20,   # Tier 2: Pinnacle CSV, medium confidence
        "xgboost": 0.25,              # Tier 3: ML prior, medium-high uncertainty
        "form_mle": 0.35,             # Form-based: lower confidence
        "league_mle": 0.50,           # Tier 4: league average, lowest confidence
    }
    ekf_P0 = ekf_P0_map.get(prediction_method, 0.25)

    # Step 7: Build Phase2Result
    return Phase2Result(
        match_id=match_id,
        league_id=league_id,
        a_H=a_H,
        a_A=a_A,
        mu_H=mu_H,
        mu_A=mu_A,
        C_time=C_time,
        verdict=verdict,
        skip_reason=skip_reason,
        param_version=param_version,
        home_team=home_team,
        away_team=away_team,
        kickoff_utc=kickoff_utc,
        kalshi_tickers=kalshi_tickers,
        market_implied=market_implied,
        prediction_method=prediction_method,
        ekf_P0=ekf_P0,
    )


async def load_production_params(config: Config, league_id: int) -> dict | None:
    """Load active production_params row for league from DB.

    Returns dict with Q, b, gamma_H, gamma_A, delta_H, delta_A, sigma_a,
    xgb_model_blob, feature_mask, version, and v5 fields.

    Backward compatible: old rows without v5 columns get sensible defaults.
    """
    dsn = (
        f"postgresql://{config.db_user}:{config.db_password}"
        f"@{config.db_host}:{config.db_port}/{config.db_name}"
    )
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:
        log.error("phase2_db_connect_failed", error=str(exc))
        return None

    try:
        row = await conn.fetchrow(
            """
            SELECT version, league_id, Q, b,
                   gamma_H, gamma_A, delta_H, delta_A,
                   sigma_a, xgb_model_blob, feature_mask,
                   trained_at, match_count, brier_score,
                   COALESCE(delta_H_pos, NULL) as delta_H_pos,
                   COALESCE(delta_H_neg, NULL) as delta_H_neg,
                   COALESCE(delta_A_pos, NULL) as delta_A_pos,
                   COALESCE(delta_A_neg, NULL) as delta_A_neg,
                   COALESCE(eta_H, 0.0) as eta_H,
                   COALESCE(eta_A, 0.0) as eta_A,
                   COALESCE(eta_H2, 0.0) as eta_H2,
                   COALESCE(eta_A2, 0.0) as eta_A2,
                   COALESCE(sigma_omega_sq, 0.01) as sigma_omega_sq
            FROM production_params
            WHERE league_id = $1 AND is_active = TRUE
            ORDER BY version DESC
            LIMIT 1
            """,
            league_id,
        )
        if row is None:
            return None

        return {
            "version": row["version"],
            "league_id": row["league_id"],
            "Q": json.loads(row["q"]),
            "b": json.loads(row["b"]),
            "gamma_H": _parse_gamma(row["gamma_h"], "home"),
            "gamma_A": _parse_gamma(row["gamma_a"], "away"),
            "delta_H": _parse_delta(row["delta_h"]),
            "delta_A": _parse_delta(row["delta_a"]),
            "sigma_a": row["sigma_a"],
            "xgb_model_blob": row["xgb_model_blob"],
            "feature_mask": json.loads(row["feature_mask"]) if row["feature_mask"] else None,
            "trained_at": row["trained_at"],
            "match_count": row["match_count"],
            "brier_score": row["brier_score"],
            # v5 fields
            "delta_H_pos": _parse_json_array(row["delta_h_pos"]),
            "delta_H_neg": _parse_json_array(row["delta_h_neg"]),
            "delta_A_pos": _parse_json_array(row["delta_a_pos"]),
            "delta_A_neg": _parse_json_array(row["delta_a_neg"]),
            "eta_H": float(row["eta_h"]),
            "eta_A": float(row["eta_a"]),
            "eta_H2": float(row["eta_h2"]),
            "eta_A2": float(row["eta_a2"]),
            "sigma_omega_sq": float(row["sigma_omega_sq"]),
        }
    finally:
        await conn.close()


def _parse_gamma(value: object, team: str) -> list[float]:
    """Parse gamma from DB — JSON string (new) or float scalar (old).

    Old schema stored a single float (gamma_H[1] or gamma_A[2]).
    New schema stores the full 4-element JSON array.
    """
    if isinstance(value, str):
        return json.loads(value)
    # Old scalar: reconstruct [0, val, -val, 0] shape (4,)
    val = float(value)
    if team == "home":
        # gamma_H was stored as gamma_H[1]; state 2 gets -val
        return [0.0, val, -val, 0.0]
    else:
        # gamma_A was stored as gamma_A[2]; state 1 gets -val
        return [0.0, -val, val, 0.0]


def _parse_delta(value: object) -> list[float]:
    """Parse delta from DB — JSON string (new) or float scalar (old).

    Old schema stored delta[2] (the ΔS=0 bin value).
    New schema stores the full 5-element JSON array.
    """
    if isinstance(value, str):
        return json.loads(value)
    # Old scalar: place the value in all non-reference bins as a rough default
    val = float(value)
    return [val, val, val, val, val]


def _parse_json_array(value: object) -> list[float] | None:
    """Parse a nullable JSON array column. Returns list or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return None


def backsolve_intensities(
    odds_implied: MarketProbs,
    b: np.ndarray,
    Q: np.ndarray,
    basis_bounds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Backsolve a_H, a_A from market-implied probabilities.

    Given P(home_win), P(draw), P(away_win), find a_H, a_A that produce
    those probabilities via the Poisson model. Uses scipy.optimize.minimize
    with an informed initial guess from implied expected goals.

    Args:
        basis_bounds: When provided, ``compute_C_time`` uses actual period
            widths instead of the 15-min default.  This ensures the
            backsolve is consistent with ``compute_remaining_mu``.

    Initial guess:
      implied_home_goals ≈ 1.5 * home_win + 1.0 * draw + 0.5 * away_win
      implied_away_goals ≈ 0.5 * home_win + 1.0 * draw + 1.5 * away_win
      a_H_0 = ln(implied_home_goals / C_time)
      a_A_0 = ln(implied_away_goals / C_time)
    """
    C_time = compute_C_time(b, basis_bounds)

    def objective(params: np.ndarray) -> float:
        a_h, a_a = params
        mu_h = np.exp(a_h) * C_time
        mu_a = np.exp(a_a) * C_time
        probs = _poisson_1x2(mu_h, mu_a)
        return float(
            (probs[0] - odds_implied.home_win) ** 2
            + (probs[1] - odds_implied.draw) ** 2
            + (probs[2] - odds_implied.away_win) ** 2
        )

    # Informed initial guess from implied expected goals
    p_h = odds_implied.home_win
    p_d = odds_implied.draw
    p_a = odds_implied.away_win
    implied_home_goals = max(0.3, 1.5 * p_h + 1.0 * p_d + 0.5 * p_a)
    implied_away_goals = max(0.3, 0.5 * p_h + 1.0 * p_d + 1.5 * p_a)
    a_H_0 = float(np.log(implied_home_goals / C_time))
    a_A_0 = float(np.log(implied_away_goals / C_time))

    x0 = np.array([a_H_0, a_A_0])
    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 2000})
    return float(result.x[0]), float(result.x[1])


def sanity_check(
    model_probs: MarketProbs,
    market_probs: MarketProbs | None,
    threshold: float = 0.15,
) -> tuple[str, str | None]:
    """Compare model vs market probabilities.

    Returns ("GO", None) or ("SKIP", "reason string").
    """
    if market_probs is None:
        return ("GO", None)

    deviations = [
        abs(model_probs.home_win - market_probs.home_win),
        abs(model_probs.draw - market_probs.draw),
        abs(model_probs.away_win - market_probs.away_win),
    ]
    max_dev = max(deviations)

    if max_dev > threshold:
        labels = ["home_win", "draw", "away_win"]
        worst = labels[deviations.index(max_dev)]
        reason = (
            f"max deviation {max_dev:.3f} > {threshold} on {worst}: "
            f"model={getattr(model_probs, worst):.3f} "
            f"market={getattr(market_probs, worst):.3f}"
        )
        log.warning("phase2_sanity_fail", reason=reason)
        return ("SKIP", reason)

    return ("GO", None)


# ─── Helpers ─────────────────────────────────────────────────


def _poisson_1x2(mu_h: float, mu_a: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Compute P(home_win), P(draw), P(away_win) from independent Poisson."""
    from scipy.stats import poisson

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, mu_h) * poisson.pmf(j, mu_a)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    return (p_home, p_draw, p_away)


def _compute_model_probs(mu_H: float, mu_A: float) -> MarketProbs:
    """Convert mu_H, mu_A to 1x2 probabilities via Poisson."""
    p_h, p_d, p_a = _poisson_1x2(mu_H, mu_A)
    return MarketProbs(home_win=p_h, draw=p_d, away_win=p_a)


def _league_mle(C_time: float) -> tuple[float, float]:
    """Default league-average log-intensities (≈1.4 home, ≈1.1 away)."""
    a_H = float(np.log(1.4 / C_time))
    a_A = float(np.log(1.1 / C_time))
    return a_H, a_A


def _build_features_from_market(
    market: MarketProbs, n_features: int | None = None,
) -> list[float]:
    """Build feature vector from market odds for XGBoost prediction.

    Adapts to the model's expected feature count (from feature_mask).
    Fills bookmaker slots with market-implied probs, team form with defaults.
    """
    h, d, a = market.home_win, market.draw, market.away_win

    if n_features is not None:
        # Compute how many 3-prob bookmaker slots fit, then add 2 form features
        n_bookie_slots = (n_features - 2) // 3 if n_features > 2 else 0
        features = [h, d, a] * n_bookie_slots + [1.4, 1.1]
        # Trim or pad to exact size
        features = features[:n_features]
        while len(features) < n_features:
            features.append(0.0)
        return features

    # Default 20-feature layout
    features = [h, d, a] * 6 + [1.4, 1.1]
    return features


async def _fetch_market_odds(
    config: Config,
    league_slug: str,
    home_team: str,
    away_team: str,
) -> MarketProbs | None:
    """Fetch pre-match odds from Odds-API and return vig-removed implied probs."""
    from src.calibration.team_aliases import normalize_team_name

    client = OddsApiClient(api_key=config.odds_api_key)
    try:
        events = await client.get_events(league_slug, status="pending")
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)

        target_event = None
        for evt in events:
            if (
                normalize_team_name(evt.get("home", "")) == home_norm
                and normalize_team_name(evt.get("away", "")) == away_norm
            ):
                target_event = evt
                break

        if target_event is None:
            log.warning("phase2_event_not_found", home=home_team, away=away_team)
            return None

        odds_data = await client.get_odds(
            target_event["id"], bookmakers="Bet365,Betfair Exchange",
        )

        return _extract_implied_probs(odds_data)
    finally:
        await client.close()


def _fetch_pinnacle_odds(
    league_id: int,
    home_team: str,
    away_team: str,
) -> MarketProbs | None:
    """Fetch Pinnacle closing odds from football-data.co.uk CSV files.

    Looks for the most recent match with matching teams in the historical
    odds data directory. Returns vig-removed implied probs or None.
    """
    from src.calibration.team_aliases import normalize_team_name

    # League ID → football-data.co.uk CSV directory name
    league_csv_dirs: dict[int, str] = {
        1204: "E0",   # EPL
        1399: "SP1",  # La Liga
        1269: "I1",   # Serie A
        1229: "D1",   # Bundesliga
        1221: "F1",   # Ligue 1
    }
    csv_dir = league_csv_dirs.get(league_id)
    if csv_dir is None:
        return None

    odds_dir = Path("data/odds_historical")
    if not odds_dir.exists():
        return None

    home_norm = normalize_team_name(home_team)
    away_norm = normalize_team_name(away_team)

    # Search CSV files (most recent season first)
    csv_files = sorted(odds_dir.glob(f"{csv_dir}*.csv"), reverse=True)
    for csv_path in csv_files:
        try:
            import csv
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_home = normalize_team_name(row.get("HomeTeam", ""))
                    row_away = normalize_team_name(row.get("AwayTeam", ""))
                    if row_home != home_norm or row_away != away_norm:
                        continue

                    # Look for Pinnacle closing odds columns
                    try:
                        ph = float(row.get("PSH") or row.get("PH", "0"))
                        pd = float(row.get("PSD") or row.get("PD", "0"))
                        pa = float(row.get("PSA") or row.get("PA", "0"))
                    except (ValueError, TypeError):
                        continue

                    if ph <= 0 or pd <= 0 or pa <= 0:
                        continue

                    p_h, p_d, p_a = _shin_vig_removal(ph, pd, pa)
                    log.info(
                        "phase2_pinnacle_found",
                        home=home_team,
                        away=away_team,
                        csv=csv_path.name,
                    )
                    return MarketProbs(
                        home_win=p_h,
                        draw=p_d,
                        away_win=p_a,
                    )
        except Exception as exc:
            log.warning("phase2_pinnacle_csv_error", csv=str(csv_path), error=str(exc))
            continue

    return None


def _extract_implied_probs(odds_data: dict) -> MarketProbs | None:
    """Extract vig-removed 1x2 implied probs from Odds-API response."""
    bookmakers = odds_data.get("bookmakers", {})
    if not bookmakers:
        return None

    # Try Bet365 first, then Betfair Exchange
    for bookie_name in ["Bet365", "Betfair Exchange"]:
        markets = bookmakers.get(bookie_name, [])
        for market in markets:
            if market.get("name") == "ML":
                odds_list = market.get("odds", [])
                if odds_list:
                    o = odds_list[0]
                    try:
                        h = float(o["home"])
                        d = float(o["draw"])
                        a = float(o["away"])
                        p_h, p_d, p_a = _shin_vig_removal(h, d, a)
                        return MarketProbs(
                            home_win=p_h,
                            draw=p_d,
                            away_win=p_a,
                        )
                    except (KeyError, ValueError, ZeroDivisionError):
                        continue
    return None


def _shin_vig_removal(
    odds_h: float, odds_d: float, odds_a: float,
) -> tuple[float, float, float]:
    """Shin method: recover true probabilities from bookmaker odds.

    Unlike naive normalization (1/odds / sum), Shin models informed
    insider trading to correctly handle favourite-longshot bias.
    Favorites get slightly higher probability; longshots get lower.

    Reference: Shin (1992, 1993), Jullien & Salanié (1994).

    Solves for z in: Σ sqrt(z² + 4(1-z)·π_i²) = 2(1-z)/O + 3z
    Then: p_i = O·(sqrt(z² + 4(1-z)·π_i²) - z) / (2(1-z))
    where π_i = (1/odds_i) / O, O = Σ(1/odds_i).
    """
    q = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    O = sum(q)
    pi = [qi / O for qi in q]

    # Bisection to find z ∈ (0, 1)
    lo, hi = 0.0, 0.99
    for _ in range(64):
        z = (lo + hi) / 2.0
        lhs = sum(math.sqrt(z * z + 4.0 * (1.0 - z) * p * p) for p in pi)
        rhs = 2.0 * (1.0 - z) / O + 3.0 * z
        if lhs > rhs:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2.0

    denom = 2.0 * (1.0 - z)
    probs = [
        O * (math.sqrt(z * z + 4.0 * (1.0 - z) * p * p) - z) / denom
        for p in pi
    ]
    return probs[0], probs[1], probs[2]


def _skip_result(
    match_id: str,
    league_id: int,
    home_team: str,
    away_team: str,
    kickoff_utc: datetime,
    reason: str,
) -> Phase2Result:
    """Build a SKIP Phase2Result with default values."""
    b = np.zeros(6)
    C_time = compute_C_time(b)
    a_H, a_A = _league_mle(C_time)
    return Phase2Result(
        match_id=match_id,
        league_id=league_id,
        a_H=a_H,
        a_A=a_A,
        mu_H=float(np.exp(a_H) * C_time),
        mu_A=float(np.exp(a_A) * C_time),
        C_time=C_time,
        verdict="SKIP",
        skip_reason=reason,
        param_version=0,
        home_team=home_team,
        away_team=away_team,
        kickoff_utc=kickoff_utc,
        kalshi_tickers={},
        market_implied=None,
        prediction_method="league_mle",
        ekf_P0=0.50,
    )
