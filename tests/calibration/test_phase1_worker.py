import pytest


@pytest.mark.slow
@pytest.mark.asyncio
async def test_phase1_epl_smoke():
    """Smoke test: run Phase 1 for EPL on real data. Should not crash.

    Run with: python -m pytest tests/calibration/test_phase1_worker.py -v -m slow
    """
    from src.calibration.phase1_worker import run_phase1
    from src.common.config import Config
    config = Config.from_env()
    result = await run_phase1("1204", config)
    assert isinstance(result, bool)
