"""Tests for EKFStrengthTracker."""

import pytest
from src.engine.ekf import EKFStrengthTracker


def test_ekf_predict_increases_uncertainty():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    P_H_before = ekf.P_H
    ekf.predict(dt=1.0)
    assert ekf.P_H > P_H_before
    assert ekf.P_H == pytest.approx(0.26, abs=0.001)


def test_ekf_goal_update_increases_scoring_team():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    a_H_before = ekf.a_H
    # Home scores with moderate intensity
    ekf.update_goal("home", lambda_H=0.03, lambda_A=0.02, dt=1.0)
    assert ekf.a_H > a_H_before  # scoring team strength increases


def test_ekf_no_goal_slightly_decreases():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    a_H_before = ekf.a_H
    # Many ticks without goals
    for _ in range(100):
        ekf.predict(dt=1.0)
        ekf.update_no_goal(lambda_H=0.03, lambda_A=0.02, dt=1.0)
    assert ekf.a_H < a_H_before  # no goals → strength drifts down


def test_ekf_surprise_score_range():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25)
    score = ekf.compute_surprise_score("away", P_model_home_win=0.70)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.70, abs=0.01)  # 1 - 0.30


def test_ekf_surprise_score_underdog():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25)
    score = ekf.compute_surprise_score("away", P_model_home_win=0.75)
    assert score > 0.5  # underdog goal is surprising


def test_ekf_clamp_prevents_divergence():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=5.0, sigma_omega_sq=0.5)
    # Extreme update that would push a_H very high
    ekf.update_goal("home", lambda_H=0.001, lambda_A=0.001, dt=1.0)
    assert ekf.a_H <= 0.3 + 1.5  # clamped
    assert ekf.a_H >= 0.3 - 1.5
