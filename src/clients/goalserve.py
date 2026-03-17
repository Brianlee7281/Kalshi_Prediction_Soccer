"""Goalserve REST client for live scores, commentaries, and upcoming fixtures."""

import html
import re
from datetime import datetime, timezone

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

log = get_logger(__name__)


class GoalserveClient:
    """Goalserve REST client for live scores and commentaries."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base = BaseClient(
            base_url=f"https://www.goalserve.com/getfeed/{api_key}/",
            timeout=20.0,
        )

    async def get_live_scores(self) -> dict:
        """GET /soccernew/home?json=1
        Returns raw JSON response with all live matches.
        """
        return await self._base.get("soccernew/home", params={"json": "1"})

    async def get_commentaries(self, league_id: str, date: str) -> dict:
        """GET /commentaries/{league_id}?date={date}&json=1
        date format: DD.MM.YYYY
        Returns raw JSON response.
        """
        return await self._base.get(
            f"commentaries/{league_id}",
            params={"date": date, "json": "1"},
        )

    def find_match_in_live(
        self, match_id: str, live_data: dict,
    ) -> dict | None:
        """Search live scores for a match by @id, @fix_id, or @static_id.
        Returns the match dict or None if not found.
        Must search ALL three ID fields (post-mortem anti-pattern fix).
        """
        categories = live_data.get("scores", {}).get("category", [])
        if isinstance(categories, dict):
            categories = [categories]

        for category in categories:
            matches = category.get("matches", {}).get("match", [])
            if isinstance(matches, dict):
                matches = [matches]
            for match in matches:
                if (
                    match.get("@id") == match_id
                    or match.get("@fix_id") == match_id
                    or match.get("@static_id") == match_id
                ):
                    return match
        return None

    async def get_upcoming_fixtures(self, league_id: str) -> list[dict]:
        """Get upcoming fixtures for a league from live scores endpoint.
        Filters matches with status = time string (e.g., "15:00", not numeric/FT/HT).
        Returns list of fixture dicts with match_id, home_team, away_team, kickoff_utc.
        """
        live_data = await self.get_live_scores()
        categories = live_data.get("scores", {}).get("category", [])
        if isinstance(categories, dict):
            categories = [categories]

        fixtures: list[dict] = []
        for category in categories:
            if category.get("@gid") != league_id and category.get("@id") != league_id:
                continue

            matches = category.get("matches", {}).get("match", [])
            if isinstance(matches, dict):
                matches = [matches]

            formatted_date = category.get("matches", {}).get("@formatted_date", "")

            for match in matches:
                status = match.get("@status", "")
                if not _is_time_string(status):
                    continue

                kickoff_utc = _parse_kickoff(formatted_date, status)
                fixtures.append({
                    "match_id": match.get("@id", ""),
                    "fix_id": match.get("@fix_id", ""),
                    "static_id": match.get("@static_id", ""),
                    "home_team": html.unescape(
                        match.get("localteam", {}).get("@name", "")
                    ),
                    "away_team": html.unescape(
                        match.get("visitorteam", {}).get("@name", "")
                    ),
                    "kickoff_utc": kickoff_utc,
                    "league_id": league_id,
                })

        log.info(
            "goalserve_upcoming",
            league_id=league_id,
            fixture_count=len(fixtures),
        )
        return fixtures

    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self._base.close()


_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _is_time_string(status: str) -> bool:
    """Check if status looks like a kickoff time (e.g. '15:00')."""
    return bool(_TIME_RE.match(status))


def _parse_kickoff(formatted_date: str, time_str: str) -> datetime:
    """Parse 'DD.MM.YYYY' + 'HH:MM' into UTC datetime."""
    try:
        dt = datetime.strptime(
            f"{formatted_date} {time_str}", "%d.%m.%Y %H:%M"
        )
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)
