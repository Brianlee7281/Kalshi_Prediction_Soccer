"""Match Goalserve fixtures to Kalshi market tickers."""

import re
from datetime import datetime

from src.calibration.team_aliases import normalize_team_name
from src.clients.kalshi import KalshiClient
from src.common.logging import get_logger

log = get_logger(__name__)


async def match_fixtures_to_tickers(
    fixtures: list[dict],
    kalshi_client: KalshiClient,
    league_prefix: str,
) -> dict[str, dict[str, str]]:
    """Match Goalserve fixtures to Kalshi tickers.

    Args:
        fixtures: [{match_id, home_team, away_team, kickoff_utc}, ...]
        kalshi_client: authenticated Kalshi client
        league_prefix: e.g. "KXEPLGAME"

    Returns:
        {match_id: {"home_win": "KXEPLGAME-...", "draw": "...", "away_win": "..."}}

    Matching logic:
    1. Fetch open markets for the series prefix
    2. For each fixture, find matching event by:
       - Team name alias matching (using normalize_team_name)
       - Accent stripping + per-word matching
       - Time window: market close_time >= fixture kickoff_utc
    3. Each match has 3 outcome markets: HOME, TIE, AWAY
    """
    markets = await kalshi_client.get_markets(league_prefix, status="open")

    # Group markets by event_ticker and extract team codes from outcome suffixes
    events: dict[str, list[dict]] = {}
    event_codes: dict[str, tuple[str, str]] = {}  # event_ticker -> (home_code, away_code)
    for m in markets:
        et = m.get("event_ticker", "")
        if et not in events:
            events[et] = []
        events[et].append(m)

    # Derive team codes from the outcome ticker suffixes (unambiguous)
    for et, mlist in events.items():
        suffixes = [t["ticker"].rsplit("-", 1)[-1] for t in mlist if "-" in t["ticker"]]
        team_suffixes = [s for s in suffixes if s != "TIE"]
        if len(team_suffixes) == 2:
            # The event ticker embeds both codes: ...{HOME}{AWAY}
            # The first code in the ticker = first team suffix found in event_ticker
            codes = _extract_teams_from_ticker(et, team_suffixes)
            if codes:
                event_codes[et] = codes

    result: dict[str, dict[str, str]] = {}

    for fixture in fixtures:
        match_id = fixture["match_id"]
        home_norm = normalize_team_name(fixture["home_team"])
        away_norm = normalize_team_name(fixture["away_team"])
        kickoff = fixture.get("kickoff_utc")

        matched_event: str | None = None

        for event_ticker, event_markets in events.items():
            # Strategy 1: match via extracted team codes
            if event_ticker in event_codes:
                home_code, away_code = event_codes[event_ticker]
                if (
                    _code_matches_team(home_code, home_norm)
                    and _code_matches_team(away_code, away_norm)
                ):
                    matched_event = event_ticker
                    break

            # Strategy 2: match via market title
            title = event_markets[0].get("title", "") if event_markets else ""
            if title and " vs " in title:
                title_home, title_away = _extract_teams_from_title(title)
                if (
                    normalize_team_name(title_home) == home_norm
                    and normalize_team_name(title_away) == away_norm
                ):
                    matched_event = event_ticker
                    break

        if matched_event is None:
            log.warning(
                "kalshi_ticker_unmatched",
                match_id=match_id,
                home=fixture["home_team"],
                away=fixture["away_team"],
            )
            continue

        # Time window check: close_time >= kickoff
        if kickoff:
            em = events[matched_event]
            close_str = em[0].get("close_time", "") if em else ""
            if close_str and isinstance(kickoff, datetime):
                try:
                    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    if close_dt < kickoff:
                        log.warning(
                            "kalshi_ticker_expired",
                            match_id=match_id,
                            close_time=close_str,
                        )
                        continue
                except (ValueError, TypeError):
                    pass

        # Classify the 3 outcome tickers using codes
        tickers: dict[str, str] = {}
        codes = event_codes.get(matched_event)
        for m in events[matched_event]:
            ticker = m["ticker"]
            suffix = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
            if suffix == "TIE":
                tickers["draw"] = ticker
            elif codes and suffix == codes[0]:
                tickers["home_win"] = ticker
            elif codes and suffix == codes[1]:
                tickers["away_win"] = ticker
            elif "home_win" not in tickers:
                tickers["home_win"] = ticker
            else:
                tickers["away_win"] = ticker

        # Verify home/away assignment
        if codes and "home_win" in tickers and "away_win" in tickers:
            home_suffix = tickers["home_win"].rsplit("-", 1)[-1]
            if not _code_matches_team(home_suffix, home_norm):
                tickers["home_win"], tickers["away_win"] = (
                    tickers["away_win"],
                    tickers["home_win"],
                )

        result[match_id] = tickers
        log.info(
            "kalshi_ticker_matched",
            match_id=match_id,
            event_ticker=matched_event,
            tickers=tickers,
        )

    return result


def _extract_teams_from_ticker(
    event_ticker: str,
    known_suffixes: list[str] | None = None,
) -> tuple[str, str] | None:
    """Parse event ticker to extract (home_code, away_code).

    Uses known outcome suffixes to determine the split point.
    e.g. KXEPLGAME-26MAR22TOTNFO with suffixes [TOT, NFO] -> ('TOT', 'NFO')

    Without suffixes, tries common 3+3 split as fallback.
    """
    parts = event_ticker.split("-", 1)
    if len(parts) < 2 or len(parts[1]) < 7:
        return None

    # Date is first 7 chars (YYMONDD), rest is concatenated team codes
    team_part = parts[1][7:]  # e.g. "TOTNFO"
    if len(team_part) < 4:
        return None

    # If we have known suffixes from outcome markets, use them to split
    if known_suffixes and len(known_suffixes) >= 2:
        for home_code in known_suffixes:
            if team_part.startswith(home_code):
                away_code = team_part[len(home_code):]
                if away_code in known_suffixes and away_code != home_code:
                    return (home_code, away_code)

    # Fallback: try 3+3, 3+2, 2+3, 3+4, 4+3
    for split_at in [3, 2, 4]:
        if split_at < len(team_part):
            home = team_part[:split_at]
            away = team_part[split_at:]
            if 2 <= len(home) <= 5 and 2 <= len(away) <= 5:
                return (home, away)

    return None


def _code_matches_team(code: str, normalized_team: str) -> bool:
    """Check if a 2-5 letter Kalshi code matches a normalized team name.

    normalized_team is already lowercased via normalize_team_name().

    Matching strategies:
    1. Code is prefix of first word (e.g. TOT -> tottenham hotspur)
    2. Code is prefix of any word (e.g. NFO -> nottingham forest)
    3. First letters of each word (e.g. WHU -> west ham united)
    4. First letter of word1 + first letters of word2 (e.g. AVL -> aston villa)
    """
    code_lower = code.lower()
    words = normalized_team.lower().split()
    if not words:
        return False

    # Direct prefix of first word
    if words[0].startswith(code_lower):
        return True

    # Direct prefix of any single word
    for w in words:
        if w.startswith(code_lower):
            return True

    # Initials of all words: W+H+U -> "whu"
    initials = "".join(w[0] for w in words)
    if initials == code_lower:
        return True

    # First letter of word1 + first N-1 letters of word2
    # e.g. AVL = A(ston) + VL(villa), NFO = N(ottingham) + FO(rest)
    if len(words) >= 2 and len(code_lower) >= 2:
        combo = words[0][0] + words[1][: len(code_lower) - 1]
        if combo == code_lower:
            return True

    # First 2 letters of word1 + first letter of word2
    # e.g. BRI = BR(ighton) + (not needed, just prefix match above)
    if len(words) >= 2 and len(code_lower) >= 3:
        combo2 = words[0][: len(code_lower) - 1] + words[1][0]
        if combo2 == code_lower:
            return True

    return False


def _extract_teams_from_title(title: str) -> tuple[str, str]:
    """Extract (home, away) from title like 'Arsenal vs Chelsea Winner?'."""
    if " vs " not in title:
        return ("", "")
    parts = title.split(" vs ", 1)
    home = parts[0].strip()
    away = parts[1].split("?")[0].strip()
    if away.lower().endswith(" winner"):
        away = away[: -len(" winner")].strip()
    return (home, away)
