"""Tests for Step 1.8: σ²_ω estimation."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.step_1_8_sigma_omega import estimate_sigma_omega_sq
from src.common.types import IntervalRecord


def _make_interval(
    match_id: str,
    t_start: float,
    t_end: float,
    delta_S: int = 0,
    home_goals: list[float] | None = None,
    away_goals: list[float] | None = None,
) -> IntervalRecord:
    return IntervalRecord(
        match_id=match_id,
        t_start=t_start,
        t_end=t_end,
        state_X=0,
        delta_S=delta_S,
        is_halftime=False,
        home_goal_times=home_goals or [],
        away_goal_times=away_goals or [],
    )


def _make_mock_opt() -> SimpleNamespace:
    return SimpleNamespace(
        delta_H=np.zeros(5),
        delta_A=np.zeros(5),
    )


def _build_matches(n_matches: int, seed: int = 42) -> dict[str, list[IntervalRecord]]:
    """Build synthetic matches with random goal patterns in both halves."""
    rng = np.random.RandomState(seed)
    intervals_by_match: dict[str, list[IntervalRecord]] = {}

    for m in range(n_matches):
        mid = f"match_{m}"
        # First half: 0-45
        first_half_goals = [float(rng.uniform(5, 40))] if rng.random() < 0.5 else []
        # Second half: 45-90
        second_half_goals = [float(rng.uniform(50, 85))] if rng.random() < 0.5 else []

        intervals_by_match[mid] = [
            _make_interval(mid, 0.0, 45.0, home_goals=first_half_goals),
            IntervalRecord(
                match_id=mid, t_start=45.0, t_end=45.0,
                state_X=0, delta_S=0, is_halftime=True,
            ),
            _make_interval(mid, 45.0, 90.0, home_goals=second_half_goals),
        ]

    return intervals_by_match


def test_sigma_omega_positive() -> None:
    """With 50+ matches, result should be positive."""
    intervals_by_match = _build_matches(60)
    result = estimate_sigma_omega_sq(intervals_by_match, _make_mock_opt())
    assert result > 0


def test_sigma_omega_default_few_matches() -> None:
    """With < 30 matches, should return default 0.01."""
    intervals_by_match = _build_matches(10)
    result = estimate_sigma_omega_sq(intervals_by_match, _make_mock_opt())
    assert result == 0.01


def test_sigma_omega_clamped() -> None:
    """Result must be in [0.001, 0.1]."""
    intervals_by_match = _build_matches(100)
    result = estimate_sigma_omega_sq(intervals_by_match, _make_mock_opt())
    assert 0.001 <= result <= 0.1
