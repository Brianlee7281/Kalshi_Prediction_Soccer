"""Tests for tick_loop v5 pipeline."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.engine.tick_loop import _sleep_until_next_tick, _compute_lambda, _basis_index


@pytest.mark.asyncio
async def test_sleep_until_next_tick() -> None:
    """Verify absolute time scheduling skips when behind."""
    start = time.monotonic()
    await _sleep_until_next_tick(start - 5.0, tick_count=1, interval=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1

    now = time.monotonic()
    await _sleep_until_next_tick(now, tick_count=1, interval=0.05)
    elapsed2 = time.monotonic() - now
    assert elapsed2 >= 0.04
    assert elapsed2 < 0.2


def test_basis_index() -> None:
    """Verify basis index lookup."""
    import numpy as np
    bounds = np.array([0, 15, 30, 47, 62, 77, 87, 92, 93])
    assert _basis_index(0.0, bounds) == 0
    assert _basis_index(14.9, bounds) == 0
    assert _basis_index(15.0, bounds) == 1
    assert _basis_index(60.0, bounds) == 3
    assert _basis_index(92.5, bounds) == 7


def test_compute_lambda() -> None:
    """Verify lambda computation from model state."""
    from src.common.types import Phase2Result
    from src.engine.model import LiveMatchModel
    import numpy as np

    result = Phase2Result(
        match_id="m1", league_id=1, a_H=0.3, a_A=0.1,
        mu_H=1.5, mu_A=1.1, C_time=1.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="league_mle",
    )
    params = {
        "Q": [[-0.02, 0.01, 0.01, 0.0], [0, -0.01, 0, 0.01],
              [0, 0, -0.01, 0.01], [0, 0, 0, 0]],
        "b": [0.1, 0.2, 0.15, 0.05, 0.1, -0.1, -0.05, -0.15],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }
    model = LiveMatchModel.from_phase2_result(result, params)
    model.t = 60.0
    model.engine_phase = "SECOND_HALF"

    lam_H = _compute_lambda(model, "home")
    lam_A = _compute_lambda(model, "away")
    assert 0.0 < lam_H < 2.0
    assert 0.0 < lam_A < 2.0
