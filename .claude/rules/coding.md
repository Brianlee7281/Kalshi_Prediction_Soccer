# Coding Conventions

## Python Style
- Python 3.11, async/await for all I/O
- Type hints on ALL function signatures (params + return)
- Pydantic BaseModel for all inter-phase data (see architecture.md §2)
- structlog for logging, never print()
- f-strings, not .format()

## Naming
- snake_case for functions, variables, files
- PascalCase for classes
- UPPER_CASE for constants
- Prefix private methods with `_`
- Match architecture.md names exactly: `P_reference`, `P_model`, `sigma_MC`, `OddsConsensusResult`

## Error Handling
- Never bare `except Exception:` — always log the error
- API calls: catch specific exceptions, log, return None or raise custom error
- Use `asyncio.wait_for(coro, timeout=N)` for all external calls

## Imports
```python
# stdlib
import asyncio
import time
# third-party
import httpx
import structlog
from pydantic import BaseModel
# local
from src.math.mc_core import run_mc_simulation
```

## Testing
- pytest + pytest-asyncio
- Test file mirrors source: `src/engine/tick_loop.py` → `tests/engine/test_tick_loop.py`
- Numerical tests: use exact expected values from architecture.md, not approx
- Fixtures in `tests/fixtures/` (JSON samples from verified API responses)

## Git
- Commit after each completed task within a sprint
- Message format: `sprint{N}: {brief description}`
- Run `make test && make lint` before commit
