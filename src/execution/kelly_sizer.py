"""Phase 4 Kelly sizing: fraction → shrinkage → surprise multiplier → dollar cap.

Pure functions — no database, no network, no API calls.
"""

from __future__ import annotations

from math import sqrt

import structlog

from src.common.types import Signal, TickPayload
from src.execution.config import CONFIG
from src.execution.signal_generator import _get_market_ekf_P, _get_market_mu

log = structlog.get_logger("kelly_sizer")


def cost_per_contract(p_kalshi: float, direction: str) -> float:
    """Cost to buy one contract: P_kalshi for YES, 1-P_kalshi for NO."""
    return p_kalshi if direction == "BUY_YES" else 1.0 - p_kalshi


def compute_kelly_fraction(
    p_model: float, p_kalshi: float, direction: str = "BUY_YES"
) -> float:
    """Raw Kelly fraction: f* = (b*p_win - q) / b, direction-aware."""
    if p_kalshi <= 0 or p_kalshi >= 1:
        return 0.0

    if direction == "BUY_YES":
        b = (1.0 / p_kalshi) - 1.0
        p_win = p_model
    else:
        b = p_kalshi / (1.0 - p_kalshi)
        p_win = 1.0 - p_model

    f_star = (b * p_win - (1.0 - p_win)) / b
    return max(0.0, f_star)


def compute_sigma_p(p_hat: float, ekf_P: float, mu_market: float) -> float:
    """Model uncertainty σ_p (same formula as signal_generator threshold)."""
    if p_hat <= 0 or p_hat >= 1:
        return 0.0

    sigma_mc_sq = p_hat * (1 - p_hat) / CONFIG.N_MC
    sigma_model_sq = ekf_P * (p_hat * (1 - p_hat) * mu_market) ** 2
    sigma_model_sq = min(sigma_model_sq, CONFIG.SIGMA_MODEL_CAP ** 2)
    return sqrt(sigma_mc_sq + sigma_model_sq)


def apply_baker_mchale_shrinkage(
    f_star: float, p_model: float, p_kalshi: float, sigma_p: float
) -> float:
    """Baker-McHale shrinkage: reduce Kelly by model uncertainty."""
    edge_sq = (p_model - p_kalshi) ** 2
    if edge_sq <= 0:
        return 0.0
    shrinkage = max(0.0, 1.0 - sigma_p**2 / edge_sq)
    return f_star * shrinkage


def apply_surprise_multiplier(f_shrunk: float, surprise_score: float) -> float:
    """SurpriseScore-adjusted Kelly multiplier (Pattern 5)."""
    kelly_mult = CONFIG.ALPHA_BASE + CONFIG.ALPHA_SURPRISE * surprise_score
    return f_shrunk * kelly_mult


def size_position(signal: Signal, payload: TickPayload, bankroll: float) -> Signal:
    """Size a signal into a tradeable position with risk caps applied."""
    f_star = compute_kelly_fraction(signal.P_model, signal.P_kalshi, signal.direction)

    mu_market = _get_market_mu(signal.market_type, payload.mu_H, payload.mu_A)
    ekf_P = _get_market_ekf_P(signal.market_type, payload.ekf_P_H, payload.ekf_P_A)
    sigma_p = compute_sigma_p(signal.P_model, ekf_P, mu_market)

    f_shrunk = apply_baker_mchale_shrinkage(
        f_star, signal.P_model, signal.P_kalshi, sigma_p
    )
    f_final = apply_surprise_multiplier(f_shrunk, payload.surprise_score)

    dollar_amount = f_final * bankroll

    # Hard caps
    dollar_amount = min(dollar_amount, CONFIG.PER_ORDER_CAP)
    dollar_amount = min(dollar_amount, CONFIG.PER_MATCH_CAP_FRAC * bankroll)

    cpc = cost_per_contract(signal.P_kalshi, signal.direction)
    contracts = int(dollar_amount / cpc) if cpc > 0 else 0

    updated = signal.model_copy(
        update={
            "kelly_fraction": f_final,
            "kelly_amount": dollar_amount,
            "contracts": contracts,
            "surprise_score": payload.surprise_score,
        }
    )

    log.info(
        "position_sized",
        ticker=signal.ticker,
        kelly=f_final,
        contracts=contracts,
    )
    return updated
