"""Tests for DomIndex."""

from src.engine.dom_index import DomIndex


def test_dom_index_home_goal():
    di = DomIndex()
    di.record_goal(30.0, "home")
    assert di.compute(30.0) > 0.0


def test_dom_index_away_goal():
    di = DomIndex()
    di.record_goal(30.0, "away")
    assert di.compute(30.0) < 0.0


def test_dom_index_decay():
    di = DomIndex()
    di.record_goal(0.0, "home")
    val_now = di.compute(0.0)
    val_later = di.compute(30.0)  # 30 min later
    assert val_later < val_now  # decayed


def test_dom_index_momentum_range():
    di = DomIndex()
    di.record_goal(30.0, "home")
    m = di.momentum_state(30.0)
    assert -1.0 < m < 1.0


def test_dom_index_quantized():
    di = DomIndex()
    assert di.quantized_state(0.0) == 0  # no goals = balanced
    di.record_goal(0.0, "home")
    assert di.quantized_state(0.0) == 1  # home dominant
