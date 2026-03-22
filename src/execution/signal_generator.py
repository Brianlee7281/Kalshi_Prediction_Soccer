"""Phase 4 signal generation: detect edges between P_model and P_kalshi.

Pure functions — no database, no network, no API calls.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import structlog

from src.common.types import MarketProbs, Signal, TickPayload
from src.execution.config import CONFIG

log = structlog.get_logger("signal_generator")

MARKET_TYPES: list[str] = ["home_win", "draw", "away_win", "over_25", "btts_yes"]


def compute_edge(p_model: float, p_kalshi: float) -> tuple[str, float]:
    """Compute EV in both directions, return best direction + EV."""
    ev_yes = p_model - p_kalshi
    ev_no = p_kalshi - p_model

    if ev_yes > ev_no and ev_yes > 0:
        direction, ev = "BUY_YES", ev_yes
    elif ev_no > ev_yes and ev_no > 0:
        direction, ev = "BUY_NO", ev_no
    else:
        direction, ev = "HOLD", 0.0

    log.debug("edge_computed", ev=ev, direction=direction)
    return direction, ev


def _get_market_mu(market_type: str, mu_H: float, mu_A: float) -> float:
    """Map market type to team-specific remaining goals."""
    if market_type == "home_win":
        return mu_H
    if market_type == "away_win":
        return mu_A
    # draw, over_25, btts_yes → conservative: larger remaining-goals estimate
    return max(mu_H, mu_A)


def _get_market_ekf_P(market_type: str, ekf_P_H: float, ekf_P_A: float) -> float:
    """Map market type to team-specific EKF uncertainty."""
    if market_type == "home_win":
        return ekf_P_H
    if market_type == "away_win":
        return ekf_P_A
    return max(ekf_P_H, ekf_P_A)


def compute_dynamic_threshold(
    p_hat: float, sigma_mc: float, ekf_P: float, mu_market: float
) -> float:
    """Dynamic edge threshold from v5 §8.2."""
    if p_hat <= 0 or p_hat >= 1:
        return 1.0

    sigma_mc_sq = p_hat * (1 - p_hat) / CONFIG.N_MC
    sigma_model_sq = ekf_P * (p_hat * (1 - p_hat) * mu_market) ** 2
    sigma_model_sq = min(sigma_model_sq, CONFIG.SIGMA_MODEL_CAP ** 2)
    sigma_p = sqrt(sigma_mc_sq + sigma_model_sq)
    theta = CONFIG.C_SPREAD + CONFIG.C_SLIPPAGE + CONFIG.Z_ALPHA * sigma_p
    return theta


def generate_signals(
    payload: TickPayload,
    p_kalshi: dict[str, float],
    kalshi_tickers: dict[str, str],
    open_positions: dict[str, Any] | None = None,
) -> list[Signal]:
    """Generate entry signals for all eligible markets.

    Returns [] immediately if payload.order_allowed is False.
    """
    if not payload.order_allowed:
        return []

    signals: list[Signal] = []

    for market_type in MARKET_TYPES:
        if market_type not in kalshi_tickers or market_type not in p_kalshi:
            continue

        # Skip markets with existing positions
        if open_positions is not None:
            has_position = any(
                getattr(pos, "market_type", None) == market_type
                for pos in open_positions.values()
            )
            if has_position:
                continue

        p_model = getattr(payload.P_model, market_type)
        if p_model is None:
            continue

        p_k = p_kalshi[market_type]

        direction, ev = compute_edge(p_model, p_k)
        if direction == "HOLD":
            continue

        mu_market = _get_market_mu(market_type, payload.mu_H, payload.mu_A)
        ekf_P = _get_market_ekf_P(market_type, payload.ekf_P_H, payload.ekf_P_A)

        sigma_mc_val = getattr(payload.sigma_MC, market_type)
        if sigma_mc_val is None:
            sigma_mc_val = 0.0

        theta = compute_dynamic_threshold(p_model, sigma_mc_val, ekf_P, mu_market)
        if ev < theta:
            continue

        signals.append(
            Signal(
                match_id=payload.match_id,
                ticker=kalshi_tickers[market_type],
                market_type=market_type,
                direction=direction,
                P_kalshi=p_k,
                P_model=p_model,
                EV=ev,
                kelly_fraction=0.0,
                kelly_amount=0.0,
                contracts=0,
                surprise_score=payload.surprise_score,
            )
        )

    log.info("signals_generated", match_id=payload.match_id, count=len(signals))
    return signals
