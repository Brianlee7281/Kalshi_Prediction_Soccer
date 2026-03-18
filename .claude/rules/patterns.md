# System Patterns — Read Every Session

## Pattern 1: P_model is Sole Authority (v5)

P_model from the 3-layer mathematical model is the ONLY probability used for trading.
No OddsConsensus, no P_reference, no signal hierarchy. Odds-API data is recorded for
post-match analysis only.

## Pattern 2: MarketProbs Decomposition (Phase 3→4)

Phase 3 emits `MarketProbs` (dict-like). Phase 4's `signal_generator` decomposes per-market:

```python
for market_type in active_markets:
    p_model = getattr(payload.P_model, market_type)      # float
    signal = generate_signal(p_model, ticker, ...)
```

NEVER pass MarketProbs directly to edge detection. Always extract the float first.

## Pattern 3: Wall-Clock Time (Phase 3 tick_loop)

```python
model.t = (time.monotonic() - kickoff_wall_clock - halftime_accumulated) / 60
```

- halftime_accumulated is updated ONCE when SECOND_HALF begins
- model.t is EFFECTIVE PLAY TIME, not wall clock
- Tick scheduling: absolute time (`next_tick = start + count * interval`), not `sleep(1)`

## Pattern 4: Reserve-Confirm-Release (Phase 4 cross-container)

```
reserve_exposure()    # DB lock <10ms, write RESERVED row
  ↓
execute_order()       # NO lock, 1-5s fill wait
  ↓
confirm_exposure()    # on fill — or release_exposure() on fail
```

CRON: release stale RESERVED entries >60s old, every 5min.

## Pattern 5: SurpriseScore-Adjusted Kelly (Phase 4, v5)

```python
# SurpriseScore = 1 - P(scoring team wins), continuous [0, 1]
# Higher = more surprising goal → higher Kelly multiplier
kelly_base = 0.10
kelly_surprise_bonus = 0.25  # max additional fraction for surprise goals
kelly_multiplier = kelly_base + kelly_surprise_bonus * payload.surprise_score
# EKF uncertainty (ekf_P_H, ekf_P_A) can further scale down when uncertain
```

## Pattern 6: Parameter Version Pinning

Running containers keep their param_version for the ENTIRE match.
`PARAMS_UPDATED` Redis message = "next match uses new version", NOT "reload now".

## Pattern 7: Recording (Sprint 3+)

All live data saved to JSONL with `_ts` (our system clock):
```
recordings/{match_id}/odds_api.jsonl    # raw WS messages
recordings/{match_id}/kalshi_ob.jsonl   # orderbook snapshot + delta
recordings/{match_id}/goalserve.jsonl   # poll responses
```

ReplayServer reads these files and replays as mock WS/HTTP. All development after Sprint 3 uses ReplayServer.

## Anti-Patterns (from 44-issue post-mortem)

- ❌ `list()` on JSON string → character-level split. Use `json.loads()`.
- ❌ `asyncio.sleep(1)` for tick timing → drift. Use absolute time scheduling.
- ❌ Single σ_MC float for all markets → wrong P_cons. Use per-market `sqrt(p*(1-p)/N)`.
- ❌ `@id` only for Goalserve match lookup → miss. Search `@id`, `@fix_id`, `@static_id`.
- ❌ Period change event every poll → spam. Track `_last_period`, emit only on change.
- ❌ No min_hold on positions → 1s churn. Always `min_hold_ticks=50`.
- ❌ No cooldown after exit → instant re-entry loop. Always `cooldown_after_exit=100`.
