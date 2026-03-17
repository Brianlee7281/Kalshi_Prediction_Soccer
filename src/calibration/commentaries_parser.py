"""Goalserve commentaries parser.

Parses local JSON files from data/commentaries/ into structured match data
for Phase 1 calibration pipeline.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def parse_minute(minute_str: str, extra_min_str: str = "") -> int:
    """Parse a minute string, handling stoppage time.

    Supports two formats:
    - Separate fields: minute="90", extra_min="5" → 95
    - Combined string: "90+5" → 95
    """
    try:
        if "+" in str(minute_str):
            parts = str(minute_str).split("+")
            return int(parts[0]) + int(parts[1])
        base = int(minute_str)
        extra = int(extra_min_str) if extra_min_str and str(extra_min_str).strip() else 0
        return base + extra
    except (ValueError, TypeError):
        return 0


def _normalize_players(players: dict | list | None) -> list[dict]:
    """Normalize player entries to always be a list."""
    if players is None:
        return []
    if isinstance(players, dict):
        return [players]
    return list(players)


def _parse_match(match: dict, league_id: str) -> dict | None:
    """Parse a single match dict into structured output."""
    status = match.get("@status", "")
    if "Full-time" not in status and status != "FT":
        return None

    summary = match.get("summary")
    if not summary:
        return None

    local = match.get("localteam", {})
    visitor = match.get("visitorteam", {})

    try:
        home_goals = int(local.get("@goals", 0))
        away_goals = int(visitor.get("@goals", 0))
    except (ValueError, TypeError):
        return None

    # Match ID: prefer @fix_id, fallback to @id
    match_id = match.get("@fix_id") or match.get("@id", "")
    if not match_id:
        return None

    # Parse goal events
    goal_events: list[dict] = []
    for team_key, team_label in [("localteam", "home"), ("visitorteam", "away")]:
        team_summary = summary.get(team_key, {})
        if not team_summary:
            continue
        goals_section = team_summary.get("goals")
        if not goals_section:
            continue
        players = _normalize_players(goals_section.get("player"))
        for p in players:
            # Skip own goals and VAR-cancelled goals
            if p.get("@owngoal", "False") == "True":
                # Own goals count for the OTHER team
                other_label = "away" if team_label == "home" else "home"
                minute = parse_minute(p.get("@minute", "0"), p.get("@extra_min", ""))
                if minute > 0:
                    goal_events.append({
                        "minute": minute,
                        "team": other_label,
                        "player": p.get("@name", ""),
                    })
                continue
            if p.get("@var_cancelled", "False") == "True":
                continue
            if p.get("@penalty_missed", "False") == "True":
                continue
            minute = parse_minute(p.get("@minute", "0"), p.get("@extra_min", ""))
            if minute > 0:
                goal_events.append({
                    "minute": minute,
                    "team": team_label,
                    "player": p.get("@name", ""),
                })

    # Parse red card events
    red_card_events: list[dict] = []
    for team_key, team_label in [("localteam", "home"), ("visitorteam", "away")]:
        team_summary = summary.get(team_key, {})
        if not team_summary:
            continue
        redcards_section = team_summary.get("redcards")
        if not redcards_section:
            continue
        players = _normalize_players(redcards_section.get("player"))
        for p in players:
            if p.get("@var_cancelled", "False") == "True":
                continue
            minute = parse_minute(p.get("@minute", "0"), p.get("@extra_min", ""))
            if minute > 0:
                red_card_events.append({
                    "minute": minute,
                    "team": team_label,
                    "player": p.get("@name", ""),
                })

    return {
        "match_id": str(match_id),
        "league_id": league_id,
        "date": match.get("@date", ""),
        "home_team": html.unescape(local.get("@name", "")),
        "away_team": html.unescape(visitor.get("@name", "")),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "goal_events": sorted(goal_events, key=lambda g: g["minute"]),
        "red_card_events": sorted(red_card_events, key=lambda r: r["minute"]),
        "status": "FT",
    }


def parse_commentaries_dir(commentaries_dir: Path) -> list[dict]:
    """Scan all JSON files in data/commentaries/.

    Returns list of match dicts with keys:
        match_id, league_id, date, home_team, away_team,
        home_goals, away_goals, goal_events, red_card_events, status
    """
    matches: list[dict] = []

    if not commentaries_dir.exists():
        log.warning("commentaries_dir_not_found", path=str(commentaries_dir))
        return matches

    for league_dir in sorted(commentaries_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league_id = league_dir.name

        for json_file in sorted(league_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("json_parse_error", file=str(json_file), error=str(e))
                continue

            # Handle list vs dict top-level format
            if isinstance(data, list):
                match_list = data
            elif isinstance(data, dict):
                # Navigate: data["commentaries"]["tournament"]["match"]
                try:
                    tournament = data["commentaries"]["tournament"]
                    match_data = tournament.get("match", [])
                    match_list = match_data if isinstance(match_data, list) else [match_data]
                except (KeyError, TypeError):
                    log.warning("unexpected_json_structure", file=str(json_file))
                    continue
            else:
                continue

            for m in match_list:
                if not isinstance(m, dict):
                    continue
                parsed = _parse_match(m, league_id)
                if parsed:
                    matches.append(parsed)

    log.info("commentaries_parsed", total_matches=len(matches))
    return matches
