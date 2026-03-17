from pathlib import Path

from src.calibration.commentaries_parser import parse_commentaries_dir, parse_minute


def test_parse_real_commentaries():
    """Parse actual data/commentaries/ and verify non-empty output."""
    matches = parse_commentaries_dir(Path("data/commentaries"))
    assert len(matches) > 100, f"Expected 100+ matches, got {len(matches)}"
    # Spot check structure
    m = matches[0]
    assert "match_id" in m
    assert "goal_events" in m
    assert "red_card_events" in m


def test_parse_goal_minute_format():
    """Verify 90+5 format is parsed as 95."""
    assert parse_minute("45") == 45
    assert parse_minute("90+3") == 93
    assert parse_minute("45+2") == 47
