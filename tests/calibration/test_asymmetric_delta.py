"""Tests for Step 1.6: asymmetric delta estimation."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.step_1_6_asymmetric_delta import estimate_asymmetric_delta
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
        gamma_H=np.zeros(4),
        gamma_A=np.zeros(4),
        b=np.zeros(6),
        a_H=np.zeros(50),
        a_A=np.zeros(50),
    )


def _build_large_intervals(n_matches: int = 10, per_match: int = 10) -> dict[str, list[IntervalRecord]]:
    """Build synthetic intervals with varying delta_S from -2 to +2."""
    rng = np.random.RandomState(42)
    intervals_by_match: dict[str, list[IntervalRecord]] = {}
    for m in range(n_matches):
        mid = f"match_{m}"
        ivs: list[IntervalRecord] = []
        for i in range(per_match):
            t_start = float(i * 9)
            t_end = t_start + 9.0
            ds = int(rng.choice([-2, -1, 0, 1, 2]))
            # Sprinkle some goals
            home_goals = [t_start + 3.0] if rng.random() < 0.3 else []
            away_goals = [t_start + 6.0] if rng.random() < 0.3 else []
            ivs.append(_make_interval(mid, t_start, t_end, delta_S=ds,
                                      home_goals=home_goals, away_goals=away_goals))
        intervals_by_match[mid] = ivs
    return intervals_by_match


def test_asymmetric_delta_output_shapes() -> None:
    intervals_by_match = _build_large_intervals(n_matches=10, per_match=10)
    mock_opt = _make_mock_opt()

    delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg = estimate_asymmetric_delta(
        intervals_by_match, mock_opt,
    )

    assert isinstance(delta_H_pos, np.ndarray)
    assert isinstance(delta_H_neg, np.ndarray)
    assert isinstance(delta_A_pos, np.ndarray)
    assert isinstance(delta_A_neg, np.ndarray)
    assert delta_H_pos.shape == (5,)
    assert delta_H_neg.shape == (5,)
    assert delta_A_pos.shape == (5,)
    assert delta_A_neg.shape == (5,)


def test_asymmetric_delta_center_bin_near_zero() -> None:
    """With balanced data at delta_S=0, bin 2 should be near 0."""
    # Create many intervals all at delta_S=0 with uniform goal rate
    intervals_by_match: dict[str, list[IntervalRecord]] = {}
    for m in range(5):
        mid = f"match_{m}"
        ivs: list[IntervalRecord] = []
        for i in range(30):
            t_start = float(i * 3)
            t_end = t_start + 3.0
            # All at delta_S=0 so they go in trailing_or_tied partition
            ivs.append(_make_interval(mid, t_start, t_end, delta_S=0,
                                      home_goals=[t_start + 1.0],
                                      away_goals=[t_start + 2.0]))
        intervals_by_match[mid] = ivs

    mock_opt = _make_mock_opt()
    _, delta_H_neg, _, delta_A_neg = estimate_asymmetric_delta(
        intervals_by_match, mock_opt,
    )

    # Bin 2 (ΔS=0) is the reference bin — log(rate/rate) = 0
    assert abs(delta_H_neg[2]) < 0.3
    assert abs(delta_A_neg[2]) < 0.3


def test_asymmetric_delta_fallback_insufficient_data() -> None:
    """With < 20 intervals per bin, should fall back to opt_result values."""
    fallback_val = 0.123
    mock_opt = SimpleNamespace(
        delta_H=np.full(5, fallback_val),
        delta_A=np.full(5, fallback_val),
    )

    # Only 5 intervals total — every bin will have < 20
    intervals_by_match = {
        "m1": [
            _make_interval("m1", 0.0, 15.0, delta_S=0),
            _make_interval("m1", 15.0, 30.0, delta_S=1),
            _make_interval("m1", 30.0, 45.0, delta_S=-1),
        ],
        "m2": [
            _make_interval("m2", 0.0, 15.0, delta_S=0),
            _make_interval("m2", 15.0, 30.0, delta_S=2),
        ],
    }

    delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg = estimate_asymmetric_delta(
        intervals_by_match, mock_opt,
    )

    # All bins should fall back to the symmetric value (clamped to [-0.5, 0.5])
    expected = np.clip(fallback_val, -0.5, 0.5)
    for arr in [delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg]:
        for val in arr:
            assert abs(val - expected) < 1e-6 or val == 0.0  # 0.0 for bins with no data and no fallback needed
