from pathlib import Path

from src.calibration.odds_loader import load_odds_csv, odds_to_implied_prob


def test_load_real_odds():
    odds = load_odds_csv(Path("data/odds_historical"))
    assert len(odds) > 500, f"Expected 500+ matches with odds, got {len(odds)}"


def test_odds_implied_probability():
    # Pinnacle odds 2.10, 3.40, 3.20 → implied probs should sum to ~1.0
    probs = odds_to_implied_prob(2.10, 3.40, 3.20)
    assert 0.98 < sum(probs) < 1.02  # after vig removal, should be ~1.0
    assert probs[0] > probs[1]  # home is favorite
