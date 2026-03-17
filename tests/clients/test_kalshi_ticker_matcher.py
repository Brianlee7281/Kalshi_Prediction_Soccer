"""Tests for Kalshi ticker matcher — ticker parsing and team code matching."""

from src.clients.kalshi_ticker_matcher import (
    _code_matches_team,
    _extract_teams_from_ticker,
    _extract_teams_from_title,
)


# ── _extract_teams_from_ticker (with known suffixes) ───────


def test_extract_teams_epl_with_suffixes():
    result = _extract_teams_from_ticker(
        "KXEPLGAME-26MAR16BREWOL", known_suffixes=["BRE", "WOL"],
    )
    assert result == ("BRE", "WOL")


def test_extract_teams_serie_a_with_suffixes():
    result = _extract_teams_from_ticker(
        "KXSERIEAGAME-26MAR15LAZACM", known_suffixes=["LAZ", "ACM"],
    )
    assert result == ("LAZ", "ACM")


def test_extract_teams_tottenham_nottingham():
    result = _extract_teams_from_ticker(
        "KXEPLGAME-26MAR22TOTNFO", known_suffixes=["TOT", "NFO"],
    )
    assert result == ("TOT", "NFO")


def test_extract_teams_aston_villa_west_ham():
    result = _extract_teams_from_ticker(
        "KXEPLGAME-26MAR22AVLWHU", known_suffixes=["AVL", "WHU"],
    )
    assert result == ("AVL", "WHU")


def test_extract_teams_bournemouth_man_utd():
    result = _extract_teams_from_ticker(
        "KXEPLGAME-26MAR20BOUMUN", known_suffixes=["BOU", "MUN"],
    )
    assert result == ("BOU", "MUN")


def test_extract_teams_fallback_no_suffixes():
    """Without suffixes, falls back to 3+3 split."""
    result = _extract_teams_from_ticker("KXEPLGAME-26MAR22TOTNFO")
    assert result is not None
    assert result == ("TOT", "NFO")


def test_extract_teams_invalid():
    assert _extract_teams_from_ticker("INVALID") is None
    assert _extract_teams_from_ticker("KXEPLGAME") is None
    assert _extract_teams_from_ticker("KXEPLGAME-26MAR") is None


# ── _code_matches_team ─────────────────────────────────────


def test_code_matches_prefix():
    """Code is prefix of first word."""
    assert _code_matches_team("TOT", "Tottenham Hotspur") is True
    assert _code_matches_team("BRE", "Brentford") is True
    assert _code_matches_team("WOL", "Wolverhampton") is True


def test_code_matches_abbreviation():
    """Multi-word teams with initial-based codes."""
    assert _code_matches_team("WHU", "West Ham United") is True
    assert _code_matches_team("NFO", "Nottingham Forest") is True


def test_code_matches_other_combos():
    """Various abbreviation patterns from real tickers."""
    assert _code_matches_team("BOU", "AFC Bournemouth") is True
    assert _code_matches_team("MUN", "Manchester United") is True
    assert _code_matches_team("ROM", "Roma") is True
    assert _code_matches_team("LEC", "Lecce") is True


def test_code_matches_normalized():
    """Works with already-lowercased normalized names."""
    assert _code_matches_team("TOT", "tottenham hotspur") is True
    assert _code_matches_team("WHU", "west ham united") is True


def test_code_no_match():
    """Code does not match team."""
    assert _code_matches_team("ARS", "Chelsea") is False
    assert _code_matches_team("XXX", "Arsenal") is False


# ── _extract_teams_from_title ──────────────────────────────


def test_extract_teams_from_title_standard():
    home, away = _extract_teams_from_title("Arsenal vs Chelsea Winner?")
    assert home == "Arsenal"
    assert away == "Chelsea"


def test_extract_teams_from_title_no_vs():
    assert _extract_teams_from_title("No versus here") == ("", "")


def test_extract_teams_from_title_roma_lecce():
    home, away = _extract_teams_from_title("Roma vs Lecce Winner?")
    assert home == "Roma"
    assert away == "Lecce"
