"""MC Pricing Bridge — connects math core MC simulation to the live model.

Runs the Numba JIT MC simulation in a thread pool executor to avoid
blocking the asyncio event loop, then converts raw results into
MarketProbs with per-market standard errors.
"""

from __future__ import annotations

import asyncio
import math
import time
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from src.common.logging import get_logger
from src.common.types import MarketProbs
from src.math.compute_mu import compute_remaining_mu
from src.math.mc_core import mc_simulate_remaining

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.mc_pricing")


async def compute_mc_prices(
    model: LiveMatchModel, N: int = 50_000
) -> tuple[MarketProbs, MarketProbs]:
    """Run MC simulation and return (P_model, sigma_MC).

    Uses src.math.mc_core.mc_simulate_remaining with current model state.
    Runs in executor (thread pool) to avoid blocking asyncio.
    Also updates model.mu_H, model.mu_A via compute_remaining_mu.
    """
    # Update remaining expected goals
    mu_H, mu_A = compute_remaining_mu(model)
    model.mu_H = mu_H
    model.mu_A = mu_A
    model.mu_H_elapsed = max(0.0, model.mu_H_at_kickoff - model.mu_H)
    model.mu_A_elapsed = max(0.0, model.mu_A_at_kickoff - model.mu_A)

    # Prepare Q decomposition for mc_simulate_remaining
    Q_diag = np.diag(model.Q).copy()
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        row_sum = 0.0
        for j in range(4):
            if i != j and model.Q[i, j] > 0:
                row_sum += model.Q[i, j]
        if row_sum > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = model.Q[i, j] / row_sum

    seed = int(time.monotonic() * 1000) % (2**31)
    S_H, S_A = model.score

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        partial(
            mc_simulate_remaining,
            t_now=model.t,
            T_end=model.T_exp,
            S_H=S_H,
            S_A=S_A,
            state=model.current_state_X,
            score_diff=model.delta_S,
            a_H=model.a_H,
            a_A=model.a_A,
            b=model.b,
            gamma_H=model.gamma_H,
            gamma_A=model.gamma_A,
            delta_H=model.delta_H,
            delta_A=model.delta_A,
            Q_diag=Q_diag,
            Q_off=Q_off,
            basis_bounds=model.basis_bounds,
            N=N,
            seed=seed,
        ),
    )

    P_model = _results_to_market_probs(results, S_H, S_A)
    sigma_MC = _compute_sigma(P_model, N)

    return P_model, sigma_MC


def _results_to_market_probs(results: np.ndarray, S_H: int, S_A: int) -> MarketProbs:
    """Convert MC simulation results (N,2) array to MarketProbs.

    results[:,0] = final home goals, results[:,1] = final away goals.
    S_H, S_A = current score (already included in results).
    """
    N = len(results)
    home_final = results[:, 0]
    away_final = results[:, 1]

    home_win = float(np.sum(home_final > away_final)) / N
    draw = float(np.sum(home_final == away_final)) / N
    away_win = float(np.sum(home_final < away_final)) / N

    total_goals = home_final + away_final
    over_25 = float(np.sum(total_goals >= 3)) / N
    under_25 = float(np.sum(total_goals < 3)) / N

    btts_yes = float(np.sum((home_final >= 1) & (away_final >= 1))) / N
    btts_no = float(np.sum((home_final < 1) | (away_final < 1))) / N

    return MarketProbs(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        over_25=over_25,
        under_25=under_25,
        btts_yes=btts_yes,
        btts_no=btts_no,
    )


def _compute_sigma(probs: MarketProbs, N: int) -> MarketProbs:
    """Per-market MC standard error: sqrt(p*(1-p)/N) for each market."""

    def _se(p: float | None) -> float | None:
        if p is None:
            return None
        return math.sqrt(p * (1.0 - p) / N)

    return MarketProbs(
        home_win=_se(probs.home_win),
        draw=_se(probs.draw),
        away_win=_se(probs.away_win),
        over_25=_se(probs.over_25),
        under_25=_se(probs.under_25),
        btts_yes=_se(probs.btts_yes),
        btts_no=_se(probs.btts_no),
    )
