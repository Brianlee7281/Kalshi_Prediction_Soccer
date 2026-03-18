# Sprint 0: Project Skeleton — Decomposition

Reference: `docs/architecture.md` §2 (Type Contracts), §5 (Data Model), §6 (Infrastructure)

## Task 0.1: pyproject.toml + Makefile

**File:** `pyproject.toml`

```toml
[project]
name = "fkt-v4"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "structlog>=24.0",
    "asyncpg>=0.29",
    "redis>=5.0",
    "numpy>=1.26",
    "numba>=0.59",
    "torch>=2.1",
    "xgboost>=2.0",
    "scipy>=1.12",
    "websockets>=12.0",
    "cryptography>=42.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "mypy>=1.8",
]
```

**File:** `Makefile`

```makefile
.PHONY: test lint up down

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

up:
	docker compose up -d

down:
	docker compose down
```

**Done:** `make lint` runs (may have 0 files), `make test` runs (may have 0 tests).

---

## Task 0.2: Docker Compose (PostgreSQL + Redis)

**File:** `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: soccer_trading
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

**File:** `docker/init.sql` — copy the FULL schema from `docs/architecture.md` §5.1.
All 8 CREATE TABLE statements + all CREATE INDEX statements. Copy exactly as written.

**Test:** `make up` → `docker compose ps` shows both running → `psql -h localhost -U postgres -d soccer_trading -c "\dt"` shows 8 tables.

---

## Task 0.3: Common Types (Pydantic Models)

**File:** `src/common/__init__.py` — empty

**File:** `src/common/types.py`

Copy ALL type contracts from `docs/architecture.md` §2 as Pydantic models. Exact models:

1. `MarketProbs` — §2.1
2. `ProductionParams` — §2.2
3. `Phase2Result` — §2.3
4. `BookmakerState` — §2.4
5. `OddsConsensusResult` — §2.4
6. `TickPayload` — §2.4
7. `Signal` — §2.5
8. `FillResult` — §2.6
9. `TickMessage` — §2.7
10. `EventMessage` — §2.7
11. `SignalMessage` — §2.7
12. `PositionUpdateMessage` — §2.7
13. `SystemAlertMessage` — §2.7

Use these imports:
```python
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel
```

Every field must match architecture.md exactly — same name, same type, same default.

**Test file:** `tests/test_types.py`

```python
from src.common.types import MarketProbs, TickPayload, Signal, OddsConsensusResult

def test_market_probs_defaults():
    mp = MarketProbs(home_win=0.4, draw=0.3, away_win=0.3)
    assert mp.over_25 is None
    assert mp.home_win + mp.draw + mp.away_win == pytest.approx(1.0)

def test_tick_payload_reference_source():
    # P_reference should exist and reference_source should be one of two values
    tp = TickPayload(
        match_id="test", t=30.0, engine_phase="FIRST_HALF",
        odds_consensus=None,
        P_model=MarketProbs(home_win=0.5, draw=0.3, away_win=0.2),
        sigma_MC=MarketProbs(home_win=0.01, draw=0.01, away_win=0.01),
        P_reference=MarketProbs(home_win=0.5, draw=0.3, away_win=0.2),
        reference_source="model",
        score=(0, 0), X=0, delta_S=0, mu_H=1.2, mu_A=0.9,
        order_allowed=True, cooldown=False, ob_freeze=False, event_state="IDLE",
    )
    assert tp.reference_source in ("consensus", "model")

def test_signal_fields():
    s = Signal(
        match_id="test", ticker="KXEPLGAME-26MAR15-HOM", market_type="home_win",
        direction="BUY_YES", P_reference=0.55, reference_source="consensus",
        P_kalshi=0.48, P_model=0.54, EV=0.07, consensus_confidence="HIGH",
        kelly_fraction=0.05, kelly_amount=25.0, contracts=52,
    )
    assert s.direction in ("BUY_YES", "BUY_NO", "HOLD")
    assert s.EV > 0

def test_odds_consensus_result():
    oc = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.5, draw=0.3, away_win=0.2),
        confidence="HIGH", n_fresh_sources=3, bookmakers=[], event_detected=False,
    )
    assert oc.confidence in ("HIGH", "LOW", "NONE")
```

**Done:** `make test` passes with 4 tests.

---

## Task 0.4: Common Logging + Config

**File:** `src/common/logging.py`

```python
import structlog

def get_logger(name: str) -> structlog.BoundLogger:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)
```

**File:** `src/common/config.py`

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "soccer_trading"
    db_user: str = "postgres"
    db_password: str = "postgres"
    redis_host: str = "localhost"
    redis_port: int = 6379
    goalserve_api_key: str = ""
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = "keys/kalshi_private.pem"
    odds_api_key: str = ""
    trading_mode: str = "paper"  # "paper" | "live"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db_host=os.environ.get("DB_HOST", "localhost"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME", "soccer_trading"),
            db_user=os.environ.get("DB_USER", "postgres"),
            db_password=os.environ.get("DB_PASSWORD", "postgres"),
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            goalserve_api_key=os.environ.get("GOALSERVE_API_KEY", ""),
            kalshi_api_key=os.environ.get("KALSHI_API_KEY", ""),
            kalshi_private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem"),
            odds_api_key=os.environ.get("ODDS_API_KEY", ""),
            trading_mode=os.environ.get("TRADING_MODE", "paper"),
        )
```

**Done:** `from src.common.logging import get_logger` and `from src.common.config import Config` work.

---

## Task 0.5: __init__.py files + Math Core Verification

Create empty `__init__.py` in every package:
- `src/__init__.py`
- `src/common/__init__.py`
- `src/math/__init__.py`
- `src/calibration/__init__.py`
- `src/clients/__init__.py`
- `src/engine/__init__.py`
- `src/execution/__init__.py`
- `src/orchestrator/__init__.py`
- `src/recorder/__init__.py`
- `src/dashboard/__init__.py`
- `tests/__init__.py`

Then verify math core imports work:

**Test file:** `tests/test_math_core.py`

```python
def test_mc_core_import():
    from src.math.mc_core import mc_simulate_remaining
    assert callable(mc_simulate_remaining)

def test_q_estimation_import():
    from src.math.step_1_2_Q_estimation import estimate_Q_global
    assert callable(estimate_Q_global)

def test_nll_import():
    from src.math.step_1_4_nll_optimize import optimize_nll, MMPPModel
    assert callable(optimize_nll)

def test_compute_mu_import():
    from src.math.compute_mu import compute_remaining_mu
    assert callable(compute_remaining_mu)
```

NOTE: These may fail if math core files have internal imports like `from src.common.types import ...` or `from src.engine.model import ...`. If so, fix the imports to work in the new directory structure. The math core files may need minor import path adjustments — change `from src.engine.xxx` to `from src.math.xxx` or add conditional imports. Do NOT change the core logic, only fix import paths.

**Done:** `make test` passes all 8 tests (4 type tests + 4 import tests).

---

## Task 0.6: .gitignore + Initial Commit

**File:** `.gitignore`

```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.env
keys/
data/commentaries/
data/odds_historical/
data/recordings/
data/feasibility/
*.db
```

After all tasks pass:

```bash
git add -A
git commit -m "sprint0: project skeleton — types, docker, makefile, math core verified"
```

**Done:** `git log` shows the commit. `make test` passes. `make up` starts postgres + redis.

---

## Execution Order

Task 0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.6

After each task, run `make test` (if tests exist) and fix any issues before proceeding.
