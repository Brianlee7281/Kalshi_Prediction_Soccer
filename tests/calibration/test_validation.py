from src.calibration.step_1_5_validation import compute_brier_score


def test_brier_score_perfect():
    # Perfect prediction: home wins, predicted 100% home
    bs = compute_brier_score([(1.0, 0.0, 0.0)], ["H"])
    assert bs == 0.0


def test_brier_score_uniform():
    # Uniform prediction for home win → standard scale ~0.222
    bs = compute_brier_score([(1 / 3, 1 / 3, 1 / 3)], ["H"])
    assert 0.20 < bs < 0.24  # should be ~0.222
