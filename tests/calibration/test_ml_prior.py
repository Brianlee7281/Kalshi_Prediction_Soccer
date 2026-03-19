import numpy as np

from src.calibration.step_1_3_ml_prior import train_xgboost_prior, compute_C_time


def test_mle_fallback():
    """With no odds data, should fall back to league MLE."""
    # Fake matches with known goal counts
    matches = [{"match_id": f"m{i}", "home_goals": 1, "away_goals": 1,
                "home_team": f"Team{i}", "away_team": f"Team{i+1}",
                "goal_events": [], "red_card_events": []} for i in range(50)]
    model, features, a_H, a_A = train_xgboost_prior(matches, {}, "1204")
    assert model is None  # no odds → MLE fallback
    assert len(a_H) == 50
    # MLE for 1 goal/match: a = ln(1/90) ≈ -4.50
    assert -5.0 < a_H[0] < -4.0


def test_C_time_default():
    C = compute_C_time(np.zeros(6))
    assert C == 90.0  # 6 periods * 15 min each (backward-compatible default)


def test_C_time_with_basis_bounds():
    """v5 8-period basis: compute_C_time uses actual period widths."""
    b = np.zeros(8)
    # basis_bounds with alpha_1=2: [0, 15, 30, 47, 62, 77, 87, 92, 93]
    # widths: 15 + 15 + 17 + 15 + 15 + 10 + 5 + 1 = 93
    bounds = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0])
    C = compute_C_time(b, basis_bounds=bounds)
    assert C == 93.0  # sum of actual period widths with exp(0)=1

    # Without basis_bounds, same 8-element b gives 120 (8 * 15)
    C_old = compute_C_time(b)
    assert C_old == 120.0
