# Sprint: run_match.py — Unified Match Runner

Single command that runs Phase 2 → waits for kickoff → starts Phase 3 recording.

```bash
PYTHONPATH=. python scripts/run_match.py --league EPL
```

## Prerequisites before running any sprint
```bash
docker compose up -d   # postgres + redis must be running
```

---

## Sprint RM-1 — Extend `run_phase2()` to return params + event_ticker

**Files touched:**
- `src/common/types.py` — add `kalshi_event_ticker` field to `Phase2Result`
- `src/prematch/phase2_pipeline.py` — derive event_ticker, return `(result, params)`
- `scripts/run_phase2.py` — unpack new tuple return, print event_ticker

---

### Prompt RM-1

```
Read these files in full before editing anything:
- src/common/types.py
- src/prematch/phase2_pipeline.py
- scripts/run_phase2.py

Make the following three targeted edits. Do not change anything else.

──────────────────────────────────────────────────
EDIT 1 — src/common/types.py

In the Phase2Result class (around the kalshi_tickers field), add one field:

  kalshi_event_ticker: str = ""   # event-level ticker, derived from kalshi_tickers

Place it directly after the kalshi_tickers field.

──────────────────────────────────────────────────
EDIT 2 — src/prematch/phase2_pipeline.py

2a. Change the return type annotation of run_phase2() from:
      -> Phase2Result
    to:
      -> tuple[Phase2Result, dict | None]

2b. At the top of run_phase2(), capture the params dict in a local variable
    called _params so we can return it. The function already calls
    load_production_params() and uses its result as `params`. Just make sure
    the variable is accessible at return time — it already is, no structural
    change needed.

2c. After Step 6 (Kalshi ticker matching), derive kalshi_event_ticker:

    # Derive event-level ticker from any market ticker
    # e.g. "KXEPLGAME-25MAR17-MANCI-HOME" → "KXEPLGAME-25MAR17-MANCI"
    kalshi_event_ticker = ""
    if kalshi_tickers:
        sample = next(iter(kalshi_tickers.values()))
        kalshi_event_ticker = sample.rsplit("-", 1)[0]

2d. In Step 7 (Build Phase2Result), add the new field:
      kalshi_event_ticker=kalshi_event_ticker,

2e. Change the final return statement from:
      return Phase2Result(...)
    to:
      return Phase2Result(...), params

    where params is the dict returned by load_production_params() earlier
    in the function. If load_production_params() returned None (no DB row),
    return (result, None).

Also update _skip_result() calls: they also return Phase2Result. Wrap each
_skip_result() return as a tuple:
  return _skip_result(...), None

──────────────────────────────────────────────────
EDIT 3 — scripts/run_phase2.py

The call to run_phase2() now returns a tuple. Unpack it:

  result, _params = await run_phase2(...)

Add one print line after the existing prints:
  if result.kalshi_event_ticker:
      print(f"  event_ticker:   {result.kalshi_event_ticker}")

No other changes to run_phase2.py.
```

---

## Sprint RM-2 — Create `scripts/run_match.py`

**Files touched:**
- `scripts/run_match.py` (new)

---

### Prompt RM-2

```
Read these files in full before writing any code:
- src/common/config.py                     (Config.from_env)
- src/common/types.py                      (Phase2Result)
- src/prematch/phase2_pipeline.py          (run_phase2 — now returns tuple,
                                            load_production_params)
- src/clients/goalserve.py                 (GoalserveClient.get_upcoming_fixtures)
- src/engine/model.py                      (LiveMatchModel.from_phase2_result)
- src/engine/tick_loop.py                  (tick_loop signature)
- src/engine/kalshi_live_poller.py         (kalshi_live_poller signature)
- src/engine/kalshi_ob_sync.py             (kalshi_ob_sync signature)
- src/recorder/recorder.py                 (MatchRecorder)
- src/clients/kalshi_ws.py                 (KalshiWSClient, disconnect())
- scripts/run_phase2.py                    (reference for style)

Create scripts/run_match.py. This is the single entry point for recording
a live match. It runs Phase 2, waits for kickoff, then starts Phase 3.

──────────────────────────────────────────────────
Usage:
  PYTHONPATH=. python scripts/run_match.py --league EPL
  PYTHONPATH=. python scripts/run_match.py --league SerieA --dry-run

──────────────────────────────────────────────────
LEAGUE_IDS = {
    "EPL": 1204, "LaLiga": 1399, "SerieA": 1269, "Bundesliga": 1229,
    "Ligue1": 1221, "MLS": 1440, "Brasileirao": 1141, "Argentina": 1081,
}

──────────────────────────────────────────────────
async def main() -> None:

  Step 1 — Parse args
    --league   str, required
    --dry-run  flag, no argument

  Step 2 — Load config
    config = Config.from_env()

  Step 3 — Find next fixture
    gs = GoalserveClient(api_key=config.goalserve_api_key)
    fixtures = await gs.get_upcoming_fixtures(str(league_id))
    await gs.close()

    If fixtures is empty:
      print(f"No upcoming fixtures for {args.league}")
      return

    fixture = fixtures[0]
    print(f"Match:    {fixture['home_team']} vs {fixture['away_team']}")
    print(f"Kickoff:  {fixture['kickoff_utc'].strftime('%Y-%m-%d %H:%M')} UTC")

  Step 4 — Run Phase 2
    result, params = await run_phase2(
        match_id=fixture["match_id"],
        league_id=league_id,
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
        kickoff_utc=fixture["kickoff_utc"],
        config=config,
    )

    Print Phase 2 summary:
      print(f"Verdict:  {result.verdict}")
      print(f"a_H/a_A:  {result.a_H:.4f} / {result.a_A:.4f}")
      print(f"mu_H/mu_A:{result.mu_H:.4f} / {result.mu_A:.4f}")
      print(f"method:   {result.prediction_method}")
      if result.kalshi_event_ticker:
          print(f"event:    {result.kalshi_event_ticker}")
      if result.kalshi_tickers:
          print(f"tickers:  {result.kalshi_tickers}")

    If result.verdict == "SKIP":
      print(f"SKIP: {result.skip_reason}")
      return

    If params is None:
      print("ERROR: No production_params in DB for this league. Run Phase 1 first.")
      return

    If --dry-run:
      print("Dry run complete.")
      return

  Step 5 — Wait for kickoff
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    seconds_until = (result.kickoff_utc - now).total_seconds()

    if seconds_until > 0:
        print(f"Waiting {seconds_until:.0f}s for kickoff...")
        await asyncio.sleep(seconds_until)
    elif seconds_until < -300:
        print(f"WARNING: kickoff was {abs(seconds_until):.0f}s ago — joining mid-game")
    else:
        print("Kickoff now — starting engine")

  Step 6 — Build model
    model = LiveMatchModel.from_phase2_result(result, params)
    model.kalshi_event_ticker = result.kalshi_event_ticker

  Step 7 — Attach recorder
    from src.recorder.recorder import MatchRecorder
    model.recorder = MatchRecorder(match_id=result.match_id)
    print(f"Recording → data/recordings/{result.match_id}/")

  Step 8 — Create WS client
    ws_client = KalshiWSClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )

  Step 9 — Run engine
    try:
        await asyncio.gather(
            kalshi_live_poller(model),
            kalshi_ob_sync(model, ws_client),
            tick_loop(model, phase4_queue=None, redis_client=None),
        )
    except asyncio.CancelledError:
        pass
    finally:
        model.recorder.finalize()
        await ws_client.disconnect()
        print(f"Done. Recordings saved → data/recordings/{result.match_id}/")

──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

──────────────────────────────────────────────────
Imports needed (use this order — stdlib, third-party, local):
  import asyncio
  import argparse
  from datetime import datetime, timezone

  from src.clients.goalserve import GoalserveClient
  from src.clients.kalshi_ws import KalshiWSClient
  from src.common.config import Config
  from src.engine.kalshi_live_poller import kalshi_live_poller
  from src.engine.kalshi_ob_sync import kalshi_ob_sync
  from src.engine.model import LiveMatchModel
  from src.engine.tick_loop import tick_loop
  from src.prematch.phase2_pipeline import run_phase2
  from src.recorder.recorder import MatchRecorder

No structlog in this script — it is a CLI entry point, use print() for
all user-facing output. Engine modules handle their own structured logging.

Type hints on main() -> None only. No other functions needed.
```

---

## Checklist

- [ ] RM-1: `Phase2Result.kalshi_event_ticker` added to types.py
- [ ] RM-1: `run_phase2()` returns `tuple[Phase2Result, dict | None]`
- [ ] RM-1: `run_phase2.py` unpacks tuple, prints event_ticker
- [ ] RM-2: `scripts/run_match.py` created
- [ ] Smoke test: `python scripts/run_match.py --league EPL --dry-run`

## After both sprints: full workflow

```bash
# 1-2 hours before kickoff — run this once and leave it running
PYTHONPATH=. python scripts/run_match.py --league EPL

# Output during Phase 2:
# Match:    Arsenal vs Chelsea
# Kickoff:  2026-03-22 15:00 UTC
# Verdict:  GO
# a_H/a_A:  0.3512 / 0.2891
# event:    KXEPLGAME-26MAR22-ARSCEL
# Waiting 4320s for kickoff...

# Output at kickoff:
# Recording → data/recordings/4346261/

# Output at FT or Ctrl+C:
# Done. Recordings saved → data/recordings/4346261/
```
