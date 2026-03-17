"""Tests for MC pricing bridge (Task 3.5)."""

from __future__ import annotations

import math

import numpy as np

from src.common.types import MarketProbs
from src.engine.mc_pricing import _compute_sigma, _results_to_market_probs


def test_results_to_market_probs() -> None:
    """Convert simulation results to MarketProbs with all market types."""
    # 600 home wins (2-1), 200 draws (1-1), 200 away wins (0-1)
    results = np.array(
        [[2, 1]] * 600 + [[1, 1]] * 200 + [[0, 1]] * 200, dtype=np.int32
    )
    probs = _results_to_market_probs(results, S_H=0, S_A=0)

    assert abs(probs.home_win - 0.60) < 0.02
    assert abs(probs.draw - 0.20) < 0.02
    assert abs(probs.away_win - 0.20) < 0.02

    # over_25: total >= 3 → home wins (2+1=3) count, draws (1+1=2) don't, away (0+1=1) don't
    assert abs(probs.over_25 - 0.60) < 0.02
    assert abs(probs.under_25 - 0.40) < 0.02

    # btts: both scored → home wins (2,1) yes, draws (1,1) yes, away (0,1) no
    assert abs(probs.btts_yes - 0.80) < 0.02
    assert abs(probs.btts_no - 0.20) < 0.02

    # Probabilities sum to 1
    assert abs(probs.home_win + probs.draw + probs.away_win - 1.0) < 0.01
    assert abs(probs.over_25 + probs.under_25 - 1.0) < 0.01
    assert abs(probs.btts_yes + probs.btts_no - 1.0) < 0.01


def test_sigma_computation() -> None:
    """Per-market MC standard error: sqrt(p*(1-p)/N)."""
    probs = MarketProbs(home_win=0.50, draw=0.30, away_win=0.20)
    sigma = _compute_sigma(probs, 50000)

    # sigma for p=0.5: sqrt(0.25/50000) ≈ 0.002236
    expected_hw = math.sqrt(0.50 * 0.50 / 50000)
    assert abs(sigma.home_win - expected_hw) < 0.0001

    # sigma for p=0.3: sqrt(0.21/50000) ≈ 0.002049
    expected_d = math.sqrt(0.30 * 0.70 / 50000)
    assert abs(sigma.draw - expected_d) < 0.0001

    # sigma for p=0.2: sqrt(0.16/50000) ≈ 0.001789
    expected_aw = math.sqrt(0.20 * 0.80 / 50000)
    assert abs(sigma.away_win - expected_aw) < 0.0001

    # None fields stay None
    assert sigma.over_25 is None
    assert sigma.btts_yes is None
