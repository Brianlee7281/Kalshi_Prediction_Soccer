"""Tests for Step 1.7: stoppage time η estimation."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.step_1_7_stoppage_eta import estimate_stoppage_eta
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


def test_eta_returns_four_floats() -> None:
    """Output should be a tuple of 4 floats."""
    # Build enough intervals to pass the minimum threshold
    intervals_by_match: dict[str, list[IntervalRecord]] = {}
    for m in range(60):
        mid = f"match_{m}"
        ivs = [
            # Normal play
            _make_interval(mid, 0.0, 44.0, home_goals=[20.0], away_goals=[30.0]),
            # 1st half stoppage (t_start >= 45, t_end <= 55)
            _make_interval(mid, 45.0, 48.0, home_goals=[46.0], away_goals=[47.0]),
            # 2nd half stoppage (t_start >= 90)
            _make_interval(mid, 90.0, 93.0, home_goals=[91.0], away_goals=[92.0]),
        ]
        intervals_by_match[mid] = ivs

    result = estimate_stoppage_eta(intervals_by_match, _make_mock_opt())

    assert isinstance(result, tuple)
    assert len(result) == 4
    for val in result:
        assert isinstance(val, float)


def test_eta_zero_insufficient_data() -> None:
    """With < 50 stoppage intervals, should return all zeros."""
    intervals_by_match = {
        "m1": [
            _make_interval("m1", 0.0, 44.0),
            _make_interval("m1", 45.0, 48.0),  # 1 stoppage interval
            _make_interval("m1", 90.0, 93.0),  # 1 stoppage interval
        ],
    }

    result = estimate_stoppage_eta(intervals_by_match, _make_mock_opt())
    assert result == (0.0, 0.0, 0.0, 0.0)


def test_eta_positive_high_stoppage_rate() -> None:
    """Stoppage intervals with 3x normal goal rate should produce eta > 0."""
    intervals_by_match: dict[str, list[IntervalRecord]] = {}

    for m in range(60):
        mid = f"match_{m}"
        # Normal play: 45 minutes, ~1 goal per 45 min each team
        normal_home = [22.0] if m % 3 == 0 else []
        normal_away = [33.0] if m % 3 == 1 else []
        ivs = [
            _make_interval(mid, 0.0, 44.0, home_goals=normal_home, away_goals=normal_away),
        ]

        # 1st half stoppage: 3 minutes, but goals at 3x rate
        # ~1 goal per 15 min = 3x the ~1 per 45 min normal rate
        stop1_home = [46.0] if m % 2 == 0 else []
        stop1_away = [47.0] if m % 2 == 1 else []
        ivs.append(_make_interval(mid, 45.0, 48.0,
                                  home_goals=stop1_home, away_goals=stop1_away))

        # 2nd half stoppage: similarly high rate
        stop2_home = [91.0] if m % 2 == 0 else []
        stop2_away = [92.0] if m % 2 == 1 else []
        ivs.append(_make_interval(mid, 90.0, 93.0,
                                  home_goals=stop2_home, away_goals=stop2_away))

        intervals_by_match[mid] = ivs

    eta_H, eta_A, eta_H2, eta_A2 = estimate_stoppage_eta(
        intervals_by_match, _make_mock_opt(),
    )

    # Stoppage rate is much higher than normal → positive eta
    assert eta_H > 0
    assert eta_H2 > 0
