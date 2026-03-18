"""Tests for Shin vig removal method."""

import math

import pytest

from src.prematch.phase2_pipeline import _shin_vig_removal


def test_shin_probabilities_sum_to_one():
    """Shin-corrected probabilities must sum to ~1.0."""
    # Typical EPL odds: home 2.10, draw 3.40, away 3.20
    p_h, p_d, p_a = _shin_vig_removal(2.10, 3.40, 3.20)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6, (
        f"Sum = {p_h + p_d + p_a:.8f}, expected 1.0"
    )


def test_shin_probabilities_positive():
    """All probabilities must be positive."""
    p_h, p_d, p_a = _shin_vig_removal(1.50, 4.00, 6.00)
    assert p_h > 0.0
    assert p_d > 0.0
    assert p_a > 0.0


def test_shin_favourite_higher_than_naive():
    """For the favourite (lowest odds), Shin gives HIGHER prob than naive."""
    odds_h, odds_d, odds_a = 1.50, 4.00, 6.00  # strong home favourite

    # Naive normalization
    raw = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    total = sum(raw)
    naive_h = raw[0] / total

    # Shin
    shin_h, _, _ = _shin_vig_removal(odds_h, odds_d, odds_a)

    assert shin_h > naive_h, (
        f"Shin({shin_h:.6f}) should be > naive({naive_h:.6f}) for favourite"
    )


def test_shin_longshot_lower_than_naive():
    """For the longshot (highest odds), Shin gives LOWER prob than naive."""
    odds_h, odds_d, odds_a = 1.50, 4.00, 6.00

    raw = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    total = sum(raw)
    naive_a = raw[2] / total

    _, _, shin_a = _shin_vig_removal(odds_h, odds_d, odds_a)

    assert shin_a < naive_a, (
        f"Shin({shin_a:.6f}) should be < naive({naive_a:.6f}) for longshot"
    )


def test_shin_fair_odds_identity():
    """When odds are already fair (sum of implied = 1.0), Shin ≈ naive."""
    # Fair odds: no vig
    p_h, p_d, p_a = _shin_vig_removal(2.0, 4.0, 4.0)
    # With fair odds: 1/2 + 1/4 + 1/4 = 1.0, so z ≈ 0
    # Shin should return approximately the naive values
    assert abs(p_h - 0.50) < 0.01
    assert abs(p_d - 0.25) < 0.01
    assert abs(p_a - 0.25) < 0.01


def test_shin_extreme_favourite():
    """Shin handles very short-priced favourite without error."""
    p_h, p_d, p_a = _shin_vig_removal(1.10, 8.00, 15.00)
    assert p_h > 0.85  # very strong favourite
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6


def test_shin_even_match():
    """Shin handles near-even odds correctly."""
    p_h, p_d, p_a = _shin_vig_removal(2.80, 3.20, 2.60)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6
    # All probs should be in reasonable range
    assert 0.20 < p_h < 0.45
    assert 0.20 < p_d < 0.40
    assert 0.25 < p_a < 0.45
