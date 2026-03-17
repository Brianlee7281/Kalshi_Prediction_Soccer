# Sprint 1: Phase 1 Calibration Pipeline — Decomposition

Reference: `docs/architecture.md` §3.1 (Phase 1), §4.1 (Goalserve), §4.4 (football-data.co.uk), §8 (Data Assets)

## Overview

Phase 1 takes historical data → trains MMPP parameters → saves to `production_params` DB table.

Pipeline: Commentaries Parser → Interval Segmentation → Q Matrix → XGBoost Prior → NLL Optimization → Validation → DB Save

Math core files already exist in `src/math/`. This sprint builds the pipeline around them.

---

## Task 1.1: IntervalRecord + Calibration Types

The math core files (`step_1_2_Q_estimation.py`, `step_1_4_nll_optimize.py`) import `from src.common.types import IntervalRecord`. This type must be added to `src/common/types.py`.

**File:** `src/common/types.py` — ADD these types (keep existing Pydantic models):

```python
from dataclasses import dataclass, field

@dataclass
class RedCardTransition:
    minute: float
    from_state: int   # 0-3
    to_state: int     # 0-3
    team: str         # "home" | "away"

@dataclass
class IntervalRecord:
    match_id: str
    t_start: float              # interval start (minutes)
    t_end: float                # interval end (minutes)
    state_X: int                # Markov state {0,1,2,3}
    delta_S: int                # score diff (home - away)
    is_halftime: bool           # True for halftime break interval
    
    # Goal events within this interval
    home_goal_times: list[float] = field(default_factory=list)
    away_goal_times: list[float] = field(default_factory=list)
    goal_delta_before: list[int] = field(default_factory=list)  # ΔS before each goal
    
    # Red card events within this interval
    red_card_transitions: list[RedCardTransition] = field(default_factory=list)
    
    # Stoppage
    alpha_1: float = 0.0        # first half stoppage minutes
```

**Test:** `tests/test_types.py` — ADD:

```python
def test_interval_record():
    from src.common.types import IntervalRecord, RedCardTransition
    ir = IntervalRecord(
        match_id="test", t_start=0.0, t_end=15.0,
        state_X=0, delta_S=0, is_halftime=False,
    )
    assert ir.home_goal_times == []
    assert ir.red_card_transitions == []

def test_interval_record_with_events():
    from src.common.types import IntervalRecord, RedCardTransition
    rc = RedCardTransition(minute=30.0, from_state=0, to_state=1, team="home")
    ir = IntervalRecord(
        match_id="test", t_start=15.0, t_end=45.0,
        state_X=0, delta_S=1, is_halftime=False,
        home_goal_times=[22.0], goal_delta_before=[0],
        red_card_transitions=[rc],
    )
    assert len(ir.red_card_transitions) == 1
    assert ir.red_card_transitions[0].to_state == 1
```

**Done:** `make test` passes (previous 8 + 2 new = 10 tests).

---

## Task 1.2: Goalserve Commentaries Parser

Parses local JSON files from `data/commentaries/` into structured match data.

**File:** `src/calibration/commentaries_parser.py`

```python
def parse_commentaries_dir(commentaries_dir: Path) -> list[dict]:
    """
    Scan all JSON files in data/commentaries/.
    Return list of match dicts with keys:
        match_id: str (goalserve @id or @fix_id)
        league_id: str
        date: str
        home_team: str
        away_team: str
        home_goals: int
        away_goals: int
        goal_events: list[{minute: int, team: "home"|"away", player: str}]
        red_card_events: list[{minute: int, team: "home"|"away", player: str}]
        status: str ("FT" for finished)
    """
```

Requirements:
- Handle both `dict` and `list` top-level JSON formats (some files are lists → skip)
- Navigate: `data["commentaries"]["tournament"]["match"]` — handle single match (dict) or multiple (list)
- Goals from: `summary.localteam.goals.player` and `summary.visitorteam.goals.player`
- Red cards from: `summary.localteam.redcards.player` and `summary.visitorteam.redcards.player`
- Handle `"90+5"` minute format: split on `+`, sum both parts
- Player entries can be dict (single) or list (multiple) — always normalize to list
- Skip matches without summary
- Use `@fix_id` as primary match_id, fallback to `@id`

**Test:** `tests/calibration/test_commentaries_parser.py`

```python
def test_parse_real_commentaries():
    """Parse actual data/commentaries/ and verify non-empty output."""
    from src.calibration.commentaries_parser import parse_commentaries_dir
    from pathlib import Path
    matches = parse_commentaries_dir(Path("data/commentaries"))
    assert len(matches) > 100, f"Expected 100+ matches, got {len(matches)}"
    # Spot check structure
    m = matches[0]
    assert "match_id" in m
    assert "goal_events" in m
    assert "red_card_events" in m

def test_parse_goal_minute_format():
    """Verify 90+5 format is parsed as 95."""
    from src.calibration.commentaries_parser import parse_minute
    assert parse_minute("45") == 45
    assert parse_minute("90+3") == 93
    assert parse_minute("45+2") == 47
```

**Done:** Parser returns 1000+ matches from local files. `make test` passes.

---

## Task 1.3: football-data.co.uk Odds Loader

Loads historical odds CSVs for XGBoost features.

**File:** `src/calibration/odds_loader.py`

```python
def load_odds_csv(odds_dir: Path) -> dict[str, dict]:
    """
    Load all CSVs from data/odds_historical/.
    Returns {match_key: {PSH, PSD, PSA, PSCH, PSCD, PSCA, B365H, B365D, B365A, ...}}
    
    match_key = "{date}_{home}_{away}" (normalized lowercase, stripped accents)
    
    European leagues (mmz4281/): have PSH (opening) + PSCH (closing) + B365 + Betfair
    Americas leagues (new/): have PSCH (closing) only + B365 + Betfair
    
    Columns to extract:
    - Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (result)
    - PSH, PSD, PSA (Pinnacle opening — Europe only)
    - PSCH, PSCD, PSCA (Pinnacle closing)
    - B365H, B365D, B365A
    - BFH, BFD, BFA (Betfair — may be missing)
    """
```

Requirements:
- Handle both European CSV format (mmz4281/) and Americas format (new/)
- Americas CSVs have different column names and include Season/Country columns
- Convert odds to implied probabilities: `p = 1/odds`, then normalize to sum=1
- Handle missing values (NaN) — skip rows without Pinnacle closing
- Strip accents from team names for matching (unicodedata.normalize)

**Test:** `tests/calibration/test_odds_loader.py`

```python
def test_load_real_odds():
    from src.calibration.odds_loader import load_odds_csv
    from pathlib import Path
    odds = load_odds_csv(Path("data/odds_historical"))
    assert len(odds) > 500, f"Expected 500+ matches with odds, got {len(odds)}"

def test_odds_implied_probability():
    from src.calibration.odds_loader import odds_to_implied_prob
    # Pinnacle odds 2.10, 3.40, 3.20 → implied probs should sum to ~1.0
    probs = odds_to_implied_prob(2.10, 3.40, 3.20)
    assert 0.98 < sum(probs) < 1.02  # after vig removal, should be ~1.0
    assert probs[0] > probs[1]  # home is favorite
```

**Done:** Loader returns 500+ matches with odds data. `make test` passes.

---

## Task 1.4: Step 1.1 — Interval Segmentation

Converts parsed match data into `IntervalRecord` list for math core consumption.

**File:** `src/calibration/step_1_1_intervals.py`

```python
def segment_match_to_intervals(match: dict) -> list[IntervalRecord]:
    """
    Convert a single parsed match into IntervalRecords.
    
    Creates intervals split at:
    - Goal events (score change → new ΔS)
    - Red card events (state change → new X)
    - Halftime (45min → halftime interval → second half)
    - 15-min basis boundaries (0, 15, 30, 45, 60, 75, 90)
    
    Each interval has constant state_X and delta_S.
    """

def segment_all_matches(matches: list[dict]) -> dict[str, list[IntervalRecord]]:
    """
    Segment all matches. Returns {match_id: [IntervalRecord, ...]}.
    Skips matches with parsing errors (log warning, don't crash).
    """
```

Requirements:
- Reconstruct Markov state path from red card events (same logic as architecture.md §3.1)
- State transitions: home red → 0→1 or 2→3, away red → 0→2 or 1→3
- Each goal creates a new interval with updated delta_S
- Halftime interval: `IntervalRecord(is_halftime=True, t_start=45, t_end=45)`
- Handle stoppage time: if goal at "45+2", t=47
- goal_delta_before tracks ΔS just before each goal (for NLL optimizer)

**Test:** `tests/calibration/test_intervals.py`

```python
def test_simple_match_no_events():
    """0-0 match with no red cards → 3 intervals (1st half, HT, 2nd half)."""
    from src.calibration.step_1_1_intervals import segment_match_to_intervals
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
    from src.calibration.step_1_1_intervals import segment_match_to_intervals
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
    from src.calibration.step_1_1_intervals import segment_match_to_intervals
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
    from pathlib import Path
    matches = parse_commentaries_dir(Path("data/commentaries"))[:50]
    intervals_by_match = segment_all_matches(matches)
    assert len(intervals_by_match) >= 40  # most should parse OK
    # Each match should have at least 3 intervals
    for mid, ivs in intervals_by_match.items():
        assert len(ivs) >= 3, f"Match {mid} has only {len(ivs)} intervals"
```

**Done:** Segmentation produces IntervalRecords. `make test` passes.

---

## Task 1.5: Step 1.2 + 1.3 — Q Matrix + XGBoost Prior

**Step 1.2** already exists in `src/math/step_1_2_Q_estimation.py`. Write a thin wrapper.

**Step 1.3** (XGBoost prior for a_H, a_A) is new code.

**File:** `src/calibration/step_1_3_ml_prior.py`

```python
def train_xgboost_prior(
    matches: list[dict],
    odds_data: dict[str, dict],
    league_id: str,
) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
    """
    Train XGBoost model to predict match-level a_H, a_A from features.
    
    Features (per match):
    - Pinnacle closing implied probs (PSCH, PSCD, PSCA) — always present
    - Pinnacle opening implied probs (PSH, PSD, PSA) — Europe only, None for Americas
    - B365 closing implied probs
    - Home/away historical goal averages (from commentaries, last 5 matches)
    
    Target: log(goals_scored / C_time) for home and away
    
    Returns:
    - xgb_model: trained XGBRegressor (or None if insufficient data)
    - feature_names: list of feature column names used
    - a_H_predictions: array of predicted a_H for each match
    - a_A_predictions: array of predicted a_A for each match
    
    Fallback (if <30 matches with odds):
    - Return None for model, use league-average MLE:
      a_H = ln(league_avg_home_goals / C_time)
      a_A = ln(league_avg_away_goals / C_time)
    """

def compute_C_time(b: np.ndarray) -> float:
    """
    Time normalization constant: C = Σ exp(b[k]) * Δτ_k
    where Δτ_k = 15 min for each of 6 basis periods.
    For initial b=[0,0,0,0,0,0], C = 6*15 = 90.
    """
```

Requirements:
- Match odds_data to commentaries by team name (accent-stripped, lowercase)
- Handle missing odds gracefully — use whatever is available
- XGBoost with `n_estimators=100, max_depth=4, learning_rate=0.1`
- 3-tier fallback: XGBoost → team form MLE (last 5 matches) → league average MLE
- Store model as `xgb_model.save_raw()` bytes for DB BYTEA column

**Test:** `tests/calibration/test_ml_prior.py`

```python
def test_mle_fallback():
    """With no odds data, should fall back to league MLE."""
    from src.calibration.step_1_3_ml_prior import train_xgboost_prior
    import numpy as np
    # Fake matches with known goal counts
    matches = [{"match_id": f"m{i}", "home_goals": 1, "away_goals": 1,
                "home_team": f"Team{i}", "away_team": f"Team{i+1}",
                "goal_events": [], "red_card_events": []} for i in range(50)]
    model, features, a_H, a_A = train_xgboost_prior(matches, {}, "1204")
    assert model is None  # no odds → MLE fallback
    assert len(a_H) == 50
    # MLE for 1 goal/match: a = ln(1/90) ≈ -4.50
    assert -5.0 < a_H[0] < -4.0

def test_C_time_default():
    from src.calibration.step_1_3_ml_prior import compute_C_time
    import numpy as np
    C = compute_C_time(np.zeros(6))
    assert C == 90.0  # 6 periods * 15 min each
```

**Done:** `make test` passes.

---

## Task 1.6: Step 1.4 + 1.5 — NLL Optimization + Validation

**Step 1.4** math core exists in `src/math/step_1_4_nll_optimize.py`. Write orchestration wrapper.

**Step 1.5** is validation with walk-forward CV.

**File:** `src/calibration/step_1_5_validation.py`

```python
def walk_forward_cv(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
    a_H_init: np.ndarray,
    a_A_init: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """
    Chronological walk-forward cross-validation.
    
    Split matches chronologically into n_folds.
    For each fold: train on all previous folds, validate on current fold.
    
    Returns:
    - per_fold_brier: list of Brier Scores per fold
    - overall_brier: mean Brier Score
    - go_nogo: "GO" if overall_brier < threshold, "NO-GO" otherwise
    """

def compute_brier_score(
    predicted_probs: list[tuple[float, float, float]],
    actual_results: list[str],
) -> float:
    """
    Brier Score for 1x2 predictions.
    predicted_probs: [(p_home, p_draw, p_away), ...]
    actual_results: ["H", "D", "A", ...]
    BS = (1/N) * Σ (p_home - y_home)² + (p_draw - y_draw)² + (p_away - y_away)²
    """
```

Requirements:
- Use `src.math.step_1_4_nll_optimize.optimize_nll` for training
- Walk-forward: fold k trains on folds 0..k-1, tests on fold k (fold 0 skipped)
- Brier Score comparison: model vs uniform (0.33, 0.33, 0.33)
- GO if model BS < uniform BS (model is better than coin flip)
- Log per-fold results with structlog

**Test:** `tests/calibration/test_validation.py`

```python
def test_brier_score_perfect():
    from src.calibration.step_1_5_validation import compute_brier_score
    # Perfect prediction: home wins, predicted 100% home
    bs = compute_brier_score([(1.0, 0.0, 0.0)], ["H"])
    assert bs == 0.0

def test_brier_score_uniform():
    from src.calibration.step_1_5_validation import compute_brier_score
    # Uniform prediction for home win
    bs = compute_brier_score([(1/3, 1/3, 1/3)], ["H"])
    assert 0.6 < bs < 0.7  # should be ~0.667
```

**Done:** `make test` passes.

---

## Task 1.7: Phase 1 Worker — Full Pipeline + DB Save

Orchestrates the entire Phase 1 pipeline and saves results to `production_params`.

**File:** `src/calibration/phase1_worker.py`

```python
async def run_phase1(league_id: str, config: Config) -> bool:
    """
    Full Phase 1 pipeline for a single league:
    
    1. Parse commentaries → matches
    2. Load odds CSV → odds_data
    3. Segment intervals → IntervalRecords
    4. Estimate Q matrix (step 1.2)
    5. Train XGBoost prior (step 1.3) → a_H_init, a_A_init
    6. NLL optimization (step 1.4) → b, gamma, delta, a_H, a_A
    7. Walk-forward validation (step 1.5) → GO/NO-GO
    8. If GO: save to production_params DB table
    
    Returns True if GO, False if NO-GO.
    """

async def save_production_params(
    config: Config,
    league_id: int,
    Q: np.ndarray,
    b: np.ndarray,
    gamma_H: float,
    gamma_A: float,
    delta_H: float,
    delta_A: float,
    sigma_a: float,
    xgb_model_blob: bytes | None,
    feature_mask: list[str] | None,
    match_count: int,
    brier_score: float,
) -> int:
    """
    INSERT into production_params table. Returns version number.
    Uses asyncpg for async DB access.
    """
```

Requirements:
- Filter commentaries by league_id
- Match commentaries to odds by team name (best-effort, log unmatched)
- If < 50 matches for a league → skip with warning
- sigma_a grid search: try [0.1, 0.3, 0.5, 1.0], pick lowest validation BS
- Save xgb_model as `model.save_raw()` bytes → BYTEA column
- Set `is_active = True` for new version, `False` for all previous versions of same league
- Log everything with structlog: match counts, BS per fold, GO/NO-GO verdict

**File:** `scripts/run_phase1.py` — CLI entry point

```python
"""
Usage: python scripts/run_phase1.py --league EPL
       python scripts/run_phase1.py --all
"""
import asyncio
import argparse
from src.calibration.phase1_worker import run_phase1
from src.common.config import Config

LEAGUE_IDS = {
    "EPL": 1204, "LaLiga": 1399, "SerieA": 1269, "Bundesliga": 1229,
    "Ligue1": 1221, "MLS": 1440, "Brasileirao": 1141, "Argentina": 1081,
}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    config = Config.from_env()
    
    if args.all:
        for name, lid in LEAGUE_IDS.items():
            print(f"\n{'='*40}\nProcessing {name} (league_id={lid})\n{'='*40}")
            result = await run_phase1(str(lid), config)
            print(f"  → {'GO' if result else 'NO-GO'}")
    elif args.league:
        lid = LEAGUE_IDS.get(args.league)
        if not lid:
            print(f"Unknown league: {args.league}. Available: {list(LEAGUE_IDS.keys())}")
            return
        result = await run_phase1(str(lid), config)
        print(f"  → {'GO' if result else 'NO-GO'}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Test:** `tests/calibration/test_phase1_worker.py`

```python
@pytest.mark.asyncio
async def test_phase1_epl_smoke():
    """Smoke test: run Phase 1 for EPL on real data. Should not crash."""
    from src.calibration.phase1_worker import run_phase1
    from src.common.config import Config
    config = Config.from_env()
    # This runs the full pipeline — may take 30-60 seconds
    result = await run_phase1("1204", config)
    assert isinstance(result, bool)
```

NOTE: This test requires Docker (postgres) to be running. Mark it with `@pytest.mark.slow` if needed.

**Done:** `python scripts/run_phase1.py --league EPL` runs, prints GO/NO-GO, and saves params to DB. `make test` passes all tests.

---

## Execution Order

Task 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7

After each task, run `make test` and fix any issues before proceeding.
Git commit after each task with message `sprint1: {brief description}`.

After Task 1.7, run the full pipeline:
```bash
make up  # ensure postgres + redis running
python scripts/run_phase1.py --league EPL
```

Sprint 1 is DONE when:
- All tests pass
- `python scripts/run_phase1.py --league EPL` prints GO
- `docker compose exec postgres psql -U postgres -d soccer_trading -c "SELECT version, league_id, match_count, brier_score, is_active FROM production_params"` shows a row
