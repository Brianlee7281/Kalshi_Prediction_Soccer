from pathlib import Path

from src.calibration.step_1_1_intervals import segment_match_to_intervals


def test_simple_match_no_events():
    """0-0 match with no red cards → 3 intervals (1st half, HT, 2nd half)."""
    match = {
        "match_id": "test1", "home_goals": 0, "away_goals": 0,
        "goal_events": [], "red_card_events": [],
    }
    intervals = segment_match_to_intervals(match)
    assert len(intervals) >= 3  # first half + halftime + second half
    assert intervals[0].state_X == 0  # 11v11
    assert intervals[0].delta_S == 0  # 0-0
    ht = [iv for iv in intervals if iv.is_halftime]
    assert len(ht) == 1


def test_match_with_goal():
    """1-0 match, goal at 30min → intervals split at minute 30."""
    match = {
        "match_id": "test2", "home_goals": 1, "away_goals": 0,
        "goal_events": [{"minute": 30, "team": "home", "player": "X"}],
        "red_card_events": [],
    }
    intervals = segment_match_to_intervals(match)
    # Before goal: delta_S=0, after goal: delta_S=1
    non_ht = [iv for iv in intervals if not iv.is_halftime]
    ds_values = [iv.delta_S for iv in non_ht]
    assert 0 in ds_values
    assert 1 in ds_values


def test_match_with_red_card():
    """Home red at 60min → state changes from 0 to 1."""
    match = {
        "match_id": "test3", "home_goals": 0, "away_goals": 0,
        "goal_events": [],
        "red_card_events": [{"minute": 60, "team": "home", "player": "Y"}],
    }
    intervals = segment_match_to_intervals(match)
    non_ht = [iv for iv in intervals if not iv.is_halftime]
    states = [iv.state_X for iv in non_ht]
    assert 0 in states  # before red
    assert 1 in states  # after red (home sent off)


def test_real_data_segmentation():
    """Segment actual commentaries data, verify reasonable output."""
    from src.calibration.commentaries_parser import parse_commentaries_dir
    from src.calibration.step_1_1_intervals import segment_all_matches

    matches = parse_commentaries_dir(Path("data/commentaries"))[:50]
    intervals_by_match = segment_all_matches(matches)
    assert len(intervals_by_match) >= 40  # most should parse OK
    # Each match should have at least 3 intervals
    for mid, ivs in intervals_by_match.items():
        assert len(ivs) >= 3, f"Match {mid} has only {len(ivs)} intervals"
