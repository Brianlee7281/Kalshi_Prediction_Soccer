"""Tests for HMMEstimator — verifies graceful degradation to DomIndex."""

from src.engine.hmm_estimator import HMMEstimator


def test_hmm_degrades_to_dom_index():
    hmm = HMMEstimator(hmm_params=None)
    assert hmm.state == 0  # no goals = balanced
    hmm.record_goal(30.0, "home")
    hmm.update(None, 30.0)
    assert hmm.state == 1  # home dominant via DomIndex


def test_hmm_adjust_intensity_passthrough():
    hmm = HMMEstimator(hmm_params=None)
    hmm.update(None, 30.0)
    # With phi=0, intensities unchanged
    lH, lA = hmm.adjust_intensity(0.03, 0.02, phi_H=0.0, phi_A=0.0)
    assert lH == 0.03
    assert lA == 0.02


def test_hmm_dom_index_value():
    hmm = HMMEstimator(hmm_params=None)
    hmm.record_goal(10.0, "home")
    hmm.update(None, 10.0)
    assert hmm.dom_index_value > 0.0
