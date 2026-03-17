"""football-data.co.uk historical odds loader.

Loads CSV files from data/odds_historical/ for Phase 1 XGBoost features.
Handles European (mmz4281/) and Americas (new/) format differences.
"""
from __future__ import annotations

import csv
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from src.calibration.team_aliases import normalize_team_name

log = structlog.get_logger(__name__)


def strip_accents(s: str) -> str:
    """Remove accent marks from a string."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _adjacent_date_keys(key: str) -> list[str]:
    """Generate match keys with ±1 day offset for timezone mismatch handling."""
    parts = key.split("_", 1)
    if len(parts) != 2:
        return []
    date_str, teams = parts
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        results = []
        for delta in [-1, 1]:
            adj = dt + timedelta(days=delta)
            results.append(f"{adj.strftime('%d/%m/%Y')}_{teams}")
        return results
    except ValueError:
        return []


def _normalize_date(date: str) -> str:
    """Normalize date separator: DD/MM/YYYY and DD.MM.YYYY both → DD/MM/YYYY."""
    return date.replace(".", "/")


def _make_match_key(date: str, home: str, away: str) -> str:
    """Create normalized match key for cross-referencing."""
    date_norm = _normalize_date(date)
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)
    return f"{date_norm}_{home_norm}_{away_norm}"


def odds_to_implied_prob(
    odds_h: float, odds_d: float, odds_a: float,
) -> tuple[float, float, float]:
    """Convert decimal odds to vig-removed implied probabilities.

    Args:
        odds_h: Home win decimal odds
        odds_d: Draw decimal odds
        odds_a: Away win decimal odds

    Returns:
        Tuple of (p_home, p_draw, p_away) summing to ~1.0
    """
    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    return (raw_h / total, raw_d / total, raw_a / total)


def _safe_float(val: str) -> float | None:
    """Parse a float, returning None for empty or invalid values."""
    if not val or not val.strip():
        return None
    try:
        v = float(val)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_european_row(row: dict) -> tuple[str, dict] | None:
    """Parse a European format CSV row (E0, SP1, D1, I1, F1)."""
    date = row.get("Date", "").strip()
    home = row.get("HomeTeam", "").strip()
    away = row.get("AwayTeam", "").strip()
    if not date or not home or not away:
        return None

    key = _make_match_key(date, home, away)

    # Pinnacle closing (always present for European)
    psch = _safe_float(row.get("PSCH", ""))
    pscd = _safe_float(row.get("PSCD", ""))
    psca = _safe_float(row.get("PSCA", ""))
    if not psch or not pscd or not psca:
        return None

    result: dict = {
        "date": date,
        "home_team": home,
        "away_team": away,
        "FTHG": _safe_float(row.get("FTHG", "")),
        "FTAG": _safe_float(row.get("FTAG", "")),
        "FTR": row.get("FTR", ""),
        # Pinnacle closing
        "PSCH": psch,
        "PSCD": pscd,
        "PSCA": psca,
    }

    # Pinnacle opening (Europe only)
    psh = _safe_float(row.get("PSH", ""))
    psd = _safe_float(row.get("PSD", ""))
    psa = _safe_float(row.get("PSA", ""))
    result["PSH"] = psh
    result["PSD"] = psd
    result["PSA"] = psa

    # B365
    result["B365H"] = _safe_float(row.get("B365H", ""))
    result["B365D"] = _safe_float(row.get("B365D", ""))
    result["B365A"] = _safe_float(row.get("B365A", ""))

    # Betfair
    result["BFH"] = _safe_float(row.get("BFH", "") or row.get("BFEH", ""))
    result["BFD"] = _safe_float(row.get("BFD", "") or row.get("BFED", ""))
    result["BFA"] = _safe_float(row.get("BFA", "") or row.get("BFEA", ""))

    # Market max & average closing
    result["MaxCH"] = _safe_float(row.get("MaxCH", "") or row.get("MaxH", ""))
    result["MaxCD"] = _safe_float(row.get("MaxCD", "") or row.get("MaxD", ""))
    result["MaxCA"] = _safe_float(row.get("MaxCA", "") or row.get("MaxA", ""))
    result["AvgCH"] = _safe_float(row.get("AvgCH", "") or row.get("AvgH", ""))
    result["AvgCD"] = _safe_float(row.get("AvgCD", "") or row.get("AvgD", ""))
    result["AvgCA"] = _safe_float(row.get("AvgCA", "") or row.get("AvgA", ""))

    return key, result


def _parse_americas_row(row: dict) -> tuple[str, dict] | None:
    """Parse an Americas format CSV row (USA, BRA, ARG)."""
    date = row.get("Date", "").strip()
    home = row.get("Home", "").strip()
    away = row.get("Away", "").strip()
    if not date or not home or not away:
        return None

    key = _make_match_key(date, home, away)

    # Pinnacle closing only for Americas
    psch = _safe_float(row.get("PSCH", ""))
    pscd = _safe_float(row.get("PSCD", ""))
    psca = _safe_float(row.get("PSCA", ""))
    if not psch or not pscd or not psca:
        return None

    result: dict = {
        "date": date,
        "home_team": home,
        "away_team": away,
        "FTHG": _safe_float(row.get("HG", "")),
        "FTAG": _safe_float(row.get("AG", "")),
        "FTR": row.get("Res", ""),
        # Pinnacle closing
        "PSCH": psch,
        "PSCD": pscd,
        "PSCA": psca,
        # No opening odds for Americas
        "PSH": None,
        "PSD": None,
        "PSA": None,
    }

    # B365 closing
    result["B365H"] = _safe_float(row.get("B365CH", ""))
    result["B365D"] = _safe_float(row.get("B365CD", ""))
    result["B365A"] = _safe_float(row.get("B365CA", ""))

    # Betfair
    result["BFH"] = _safe_float(row.get("BFECH", ""))
    result["BFD"] = _safe_float(row.get("BFECD", ""))
    result["BFA"] = _safe_float(row.get("BFECA", ""))

    # Market max & average closing
    result["MaxCH"] = _safe_float(row.get("MaxCH", ""))
    result["MaxCD"] = _safe_float(row.get("MaxCD", ""))
    result["MaxCA"] = _safe_float(row.get("MaxCA", ""))
    result["AvgCH"] = _safe_float(row.get("AvgCH", ""))
    result["AvgCD"] = _safe_float(row.get("AvgCD", ""))
    result["AvgCA"] = _safe_float(row.get("AvgCA", ""))

    return key, result


# Americas CSV filenames
_AMERICAS_FILES = {"USA.csv", "BRA.csv", "ARG.csv"}


def load_odds_csv(odds_dir: Path) -> dict[str, dict]:
    """Load all CSVs from data/odds_historical/.

    Returns:
        Dict mapping match_key ("{date}_{home}_{away}" normalized) to odds dict.
    """
    result: dict[str, dict] = {}

    if not odds_dir.exists():
        log.warning("odds_dir_not_found", path=str(odds_dir))
        return result

    for csv_file in sorted(odds_dir.glob("*.csv")):
        is_americas = csv_file.name in _AMERICAS_FILES

        try:
            with open(csv_file, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if is_americas:
                            parsed = _parse_americas_row(row)
                        else:
                            parsed = _parse_european_row(row)

                        if parsed:
                            key, data = parsed
                            result[key] = data
                            # Also index with ±1 day for timezone mismatches
                            # (Americas leagues: evening matches cross midnight UTC)
                            for adj_key in _adjacent_date_keys(key):
                                if adj_key not in result:
                                    result[adj_key] = data
                    except Exception:
                        continue
        except (OSError, UnicodeDecodeError) as e:
            log.warning("csv_read_error", file=str(csv_file), error=str(e))
            continue

    log.info("odds_loaded", total_matches=len(result))
    return result
