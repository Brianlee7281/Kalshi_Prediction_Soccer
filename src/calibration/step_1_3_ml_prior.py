"""Step 1.3 — ML Prior for match-level intensities.

Trains XGBoost model to predict match-level a_H, a_A from odds features.
3-tier fallback: XGBoost → team form MLE (last 5 matches) → league average MLE.
"""
from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import structlog

from src.calibration.team_aliases import normalize_team_name

log = structlog.get_logger(__name__)


def compute_C_time(
    b: np.ndarray,
    basis_bounds: np.ndarray | None = None,
) -> float:
    """Time normalization constant: C = sum(exp(b[k]) * width[k]).

    When *basis_bounds* is provided and its length matches ``len(b) + 1``,
    each period width is taken from ``np.diff(basis_bounds)``.  This is the
    correct calculation for the v5 8-period basis where periods have variable
    widths (e.g. [15, 15, 17, 15, 15, 10, 5, 1] totalling 93 min).

    When *basis_bounds* is ``None`` (backward-compatible default), every
    period is assumed to be 15 minutes wide.  With 6 elements this gives
    C = 6 * 15 = 90, matching the original v4 convention.
    """
    if basis_bounds is not None and len(basis_bounds) == len(b) + 1:
        widths = np.diff(basis_bounds)
        return float(np.sum(np.exp(b) * widths))
    delta_tau = 15.0
    return float(np.sum(np.exp(b) * delta_tau))


def _strip_accents(s: str) -> str:
    """Remove accent marks from a string."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _match_key_from_commentary(match: dict) -> str:
    """Build a normalized key for matching commentaries to odds."""
    home = normalize_team_name(match.get("home_team", ""))
    away = normalize_team_name(match.get("away_team", ""))
    date = match.get("date", "").replace(".", "/")
    return f"{date}_{home}_{away}"


def _build_team_form(
    matches: list[dict],
) -> dict[str, list[tuple[int, int]]]:
    """Build per-team goal history: {team_name: [(goals_for, goals_against), ...]}."""
    form: dict[str, list[tuple[int, int]]] = {}
    for m in matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        hg = m.get("home_goals", 0)
        ag = m.get("away_goals", 0)
        form.setdefault(home, []).append((hg, ag))
        form.setdefault(away, []).append((ag, hg))
    return form


def _team_form_mle(
    form: dict[str, list[tuple[int, int]]],
    home_team: str,
    away_team: str,
    C_time: float,
    n_recent: int = 5,
) -> tuple[float, float] | None:
    """Compute a_H, a_A from team's recent form (last n matches).

    Returns None if either team has no history.
    """
    home_hist = form.get(home_team, [])
    away_hist = form.get(away_team, [])
    if not home_hist or not away_hist:
        return None

    home_recent = home_hist[-n_recent:]
    away_recent = away_hist[-n_recent:]

    avg_home_goals = np.mean([g[0] for g in home_recent])
    avg_away_goals = np.mean([g[0] for g in away_recent])

    # Clamp to avoid log(0)
    avg_home_goals = max(avg_home_goals, 0.1)
    avg_away_goals = max(avg_away_goals, 0.1)

    a_H = float(np.log(avg_home_goals / C_time))
    a_A = float(np.log(avg_away_goals / C_time))
    return a_H, a_A


def _odds_to_implied(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Convert decimal odds to vig-removed implied probabilities."""
    raw_h = 1.0 / h
    raw_d = 1.0 / d
    raw_a = 1.0 / a
    total = raw_h + raw_d + raw_a
    return (raw_h / total, raw_d / total, raw_a / total)


def train_xgboost_prior(
    matches: list[dict],
    odds_data: dict[str, dict],
    league_id: str,
) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
    """Train XGBoost model to predict match-level a_H, a_A from features.

    Features (per match):
    - Pinnacle closing implied probs (PSCH, PSCD, PSCA) — always present
    - Pinnacle opening implied probs (PSH, PSD, PSA) — Europe only, None for Americas
    - B365 closing implied probs
    - Home/away historical goal averages (from commentaries, last 5 matches)

    Target: log(goals_scored / C_time) for home and away

    Returns:
        xgb_model: trained XGBRegressor (or None if insufficient data)
        feature_names: list of feature column names used
        a_H_predictions: array of predicted a_H for each match
        a_A_predictions: array of predicted a_A for each match

    Fallback (if <30 matches with odds):
        Return None for model, use team form MLE or league-average MLE.
    """
    n_matches = len(matches)
    b_init = np.zeros(6)
    C = compute_C_time(b_init)

    # Build team form for tier-2 fallback
    team_form = _build_team_form(matches)

    # Compute league averages for tier-3 fallback
    total_home = sum(m.get("home_goals", 0) for m in matches)
    total_away = sum(m.get("away_goals", 0) for m in matches)
    avg_home = max(total_home / max(n_matches, 1), 0.1)
    avg_away = max(total_away / max(n_matches, 1), 0.1)
    league_a_H = float(np.log(avg_home / C))
    league_a_A = float(np.log(avg_away / C))

    # Try to match commentaries to odds
    matched_indices: list[int] = []
    features_list: list[list[float]] = []
    targets_H: list[float] = []
    targets_A: list[float] = []

    for i, m in enumerate(matches):
        key = _match_key_from_commentary(m)
        odds = odds_data.get(key)
        if odds is None:
            # Try swapped home/away
            home = normalize_team_name(m.get("home_team", ""))
            away = normalize_team_name(m.get("away_team", ""))
            date = m.get("date", "").replace(".", "/")
            key_swap = f"{date}_{away}_{home}"
            odds = odds_data.get(key_swap)
        if odds is None:
            continue

        psch = odds.get("PSCH")
        pscd = odds.get("PSCD")
        psca = odds.get("PSCA")
        if not psch or not pscd or not psca:
            continue

        # Build feature vector
        ps_closing = _odds_to_implied(psch, pscd, psca)
        row: list[float] = [ps_closing[0], ps_closing[1], ps_closing[2]]

        # Opening odds (Europe only)
        psh = odds.get("PSH")
        psd = odds.get("PSD")
        psa = odds.get("PSA")
        if psh and psd and psa:
            ps_opening = _odds_to_implied(psh, psd, psa)
            row.extend([ps_opening[0], ps_opening[1], ps_opening[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # B365 odds
        b365h = odds.get("B365H")
        b365d = odds.get("B365D")
        b365a = odds.get("B365A")
        if b365h and b365d and b365a:
            b365_impl = _odds_to_implied(b365h, b365d, b365a)
            row.extend([b365_impl[0], b365_impl[1], b365_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # Betfair exchange odds
        bfh = odds.get("BFH")
        bfd = odds.get("BFD")
        bfa = odds.get("BFA")
        if bfh and bfd and bfa:
            bf_impl = _odds_to_implied(bfh, bfd, bfa)
            row.extend([bf_impl[0], bf_impl[1], bf_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # Market max closing odds
        maxh = odds.get("MaxCH")
        maxd = odds.get("MaxCD")
        maxa = odds.get("MaxCA")
        if maxh and maxd and maxa:
            max_impl = _odds_to_implied(maxh, maxd, maxa)
            row.extend([max_impl[0], max_impl[1], max_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # Market average closing odds
        avgh = odds.get("AvgCH")
        avgd = odds.get("AvgCD")
        avga = odds.get("AvgCA")
        if avgh and avgd and avga:
            avg_impl = _odds_to_implied(avgh, avgd, avga)
            row.extend([avg_impl[0], avg_impl[1], avg_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # Team form features
        home_hist = team_form.get(m.get("home_team", ""), [])
        away_hist = team_form.get(m.get("away_team", ""), [])
        recent_home = home_hist[-5:] if home_hist else []
        recent_away = away_hist[-5:] if away_hist else []
        avg_home_f = np.mean([g[0] for g in recent_home]) if recent_home else avg_home
        avg_away_f = np.mean([g[0] for g in recent_away]) if recent_away else avg_away
        row.extend([float(avg_home_f), float(avg_away_f)])

        matched_indices.append(i)
        features_list.append(row)

        hg = max(m.get("home_goals", 0), 0.1)
        ag = max(m.get("away_goals", 0), 0.1)
        targets_H.append(float(np.log(hg / C)))
        targets_A.append(float(np.log(ag / C)))

    feature_names = [
        "ps_close_h", "ps_close_d", "ps_close_a",
        "ps_open_h", "ps_open_d", "ps_open_a",
        "b365_h", "b365_d", "b365_a",
        "bf_h", "bf_d", "bf_a",
        "max_h", "max_d", "max_a",
        "avg_h", "avg_d", "avg_a",
        "form_home_avg", "form_away_avg",
    ]

    # Tier 1: XGBoost (if enough matched data)
    if len(matched_indices) >= 30:
        try:
            import xgboost as xgb

            X = np.array(features_list)
            y_H = np.array(targets_H)
            y_A = np.array(targets_A)

            model_H = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                verbosity=0,
            )
            model_A = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                verbosity=0,
            )
            model_H.fit(X, y_H)
            model_A.fit(X, y_A)

            # Predict for ALL matches (matched get XGB, unmatched get fallback)
            a_H_all = np.full(n_matches, league_a_H)
            a_A_all = np.full(n_matches, league_a_A)

            pred_H = model_H.predict(X)
            pred_A = model_A.predict(X)
            for j, idx in enumerate(matched_indices):
                a_H_all[idx] = pred_H[j]
                a_A_all[idx] = pred_A[j]

            # Fill unmatched with team form MLE or league MLE
            matched_set = set(matched_indices)
            for i, m in enumerate(matches):
                if i in matched_set:
                    continue
                form_result = _team_form_mle(
                    team_form, m.get("home_team", ""), m.get("away_team", ""), C,
                )
                if form_result:
                    a_H_all[i], a_A_all[i] = form_result

            log.info(
                "xgboost_prior_trained",
                league_id=league_id,
                matched=len(matched_indices),
                total=n_matches,
            )
            return (model_H, model_A), feature_names, a_H_all, a_A_all

        except ImportError:
            log.warning("xgboost_not_available", fallback="mle")
        except Exception as e:
            log.warning("xgboost_training_failed", error=str(e), fallback="mle")

    # Tier 2 & 3: Team form MLE → league average MLE
    log.info(
        "mle_fallback",
        league_id=league_id,
        matched_odds=len(matched_indices),
        threshold=30,
    )
    a_H_all = np.full(n_matches, league_a_H)
    a_A_all = np.full(n_matches, league_a_A)

    for i, m in enumerate(matches):
        form_result = _team_form_mle(
            team_form, m.get("home_team", ""), m.get("away_team", ""), C,
        )
        if form_result:
            a_H_all[i], a_A_all[i] = form_result

    return None, feature_names, a_H_all, a_A_all


def predict_xgboost_prior(
    xgb_models: tuple[Any, Any],
    matches: list[dict],
    odds_data: dict[str, dict],
    train_matches: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a_H, a_A for new matches using a pre-trained XGBoost model.

    Uses train_matches for team form and league averages (no leakage from
    the prediction set). Falls back to team-form MLE or league MLE for
    matches without odds data.

    Args:
        xgb_models: (model_H, model_A) from train_xgboost_prior.
        matches: Matches to predict for (e.g. validation fold).
        odds_data: Full odds dict.
        train_matches: Training fold matches (for form/league avg context).

    Returns:
        (a_H_array, a_A_array) for each match in matches.
    """
    model_H, model_A = xgb_models
    n = len(matches)
    b_init = np.zeros(6)
    C = compute_C_time(b_init)

    # Build team form from TRAINING data only
    team_form = _build_team_form(train_matches)

    # League averages from TRAINING data only
    total_home = sum(m.get("home_goals", 0) for m in train_matches)
    total_away = sum(m.get("away_goals", 0) for m in train_matches)
    n_train = max(len(train_matches), 1)
    avg_home = max(total_home / n_train, 0.1)
    avg_away = max(total_away / n_train, 0.1)
    league_a_H = float(np.log(avg_home / C))
    league_a_A = float(np.log(avg_away / C))

    a_H_all = np.full(n, league_a_H)
    a_A_all = np.full(n, league_a_A)

    # Build features for matches with odds
    matched_indices: list[int] = []
    features_list: list[list[float]] = []

    for i, m in enumerate(matches):
        key = _match_key_from_commentary(m)
        odds = odds_data.get(key)
        if odds is None:
            home = normalize_team_name(m.get("home_team", ""))
            away = normalize_team_name(m.get("away_team", ""))
            date = m.get("date", "").replace(".", "/")
            key_swap = f"{date}_{away}_{home}"
            odds = odds_data.get(key_swap)
        if odds is None:
            # Fallback: team form MLE
            form_result = _team_form_mle(
                team_form, m.get("home_team", ""), m.get("away_team", ""), C,
            )
            if form_result:
                a_H_all[i], a_A_all[i] = form_result
            continue

        psch = odds.get("PSCH")
        pscd = odds.get("PSCD")
        psca = odds.get("PSCA")
        if not psch or not pscd or not psca:
            form_result = _team_form_mle(
                team_form, m.get("home_team", ""), m.get("away_team", ""), C,
            )
            if form_result:
                a_H_all[i], a_A_all[i] = form_result
            continue

        ps_closing = _odds_to_implied(psch, pscd, psca)
        row: list[float] = [ps_closing[0], ps_closing[1], ps_closing[2]]

        psh = odds.get("PSH")
        psd = odds.get("PSD")
        psa = odds.get("PSA")
        if psh and psd and psa:
            ps_opening = _odds_to_implied(psh, psd, psa)
            row.extend([ps_opening[0], ps_opening[1], ps_opening[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        b365h = odds.get("B365H")
        b365d = odds.get("B365D")
        b365a = odds.get("B365A")
        if b365h and b365d and b365a:
            b365_impl = _odds_to_implied(b365h, b365d, b365a)
            row.extend([b365_impl[0], b365_impl[1], b365_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        bfh = odds.get("BFH")
        bfd = odds.get("BFD")
        bfa = odds.get("BFA")
        if bfh and bfd and bfa:
            bf_impl = _odds_to_implied(bfh, bfd, bfa)
            row.extend([bf_impl[0], bf_impl[1], bf_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        maxh = odds.get("MaxCH")
        maxd = odds.get("MaxCD")
        maxa = odds.get("MaxCA")
        if maxh and maxd and maxa:
            max_impl = _odds_to_implied(maxh, maxd, maxa)
            row.extend([max_impl[0], max_impl[1], max_impl[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        avgh = odds.get("AvgCH")
        avgd = odds.get("AvgCD")
        avga = odds.get("AvgCA")
        if avgh and avgd and avga:
            avg_impl_val = _odds_to_implied(avgh, avgd, avga)
            row.extend([avg_impl_val[0], avg_impl_val[1], avg_impl_val[2]])
        else:
            row.extend([ps_closing[0], ps_closing[1], ps_closing[2]])

        # Team form from TRAINING data
        home_hist = team_form.get(m.get("home_team", ""), [])
        away_hist = team_form.get(m.get("away_team", ""), [])
        recent_home = home_hist[-5:] if home_hist else []
        recent_away = away_hist[-5:] if away_hist else []
        avg_home_f = np.mean([g[0] for g in recent_home]) if recent_home else avg_home
        avg_away_f = np.mean([g[0] for g in recent_away]) if recent_away else avg_away
        row.extend([float(avg_home_f), float(avg_away_f)])

        matched_indices.append(i)
        features_list.append(row)

    if features_list:
        X = np.array(features_list)
        pred_H = model_H.predict(X)
        pred_A = model_A.predict(X)
        for j, idx in enumerate(matched_indices):
            a_H_all[idx] = pred_H[j]
            a_A_all[idx] = pred_A[j]

    return a_H_all, a_A_all
