# CLAUDE.md — MMPP Soccer Live Trading System (v5)

Cross-market soccer trading: 3-layer MMPP model → edge vs Kalshi → automated execution.

## Architecture (4 phases per match)

1. **Phase 1** (offline weekly): Train MMPP params → `production_params` table
2. **Phase 2** (kickoff −65min): Backsolve intensities → GO/SKIP
3. **Phase 3** (live 90min): EKF + HMM/DomIndex + MC simulation → P_model/sec
4. **Phase 4** (live 90min): Edge = P_model − P_kalshi → Kelly → Kalshi orders

Infra: Docker (1 container/match), PostgreSQL, Redis, FastAPI + React dashboard.

## Read Before Working

| Working in | Read first |
|---|---|
| `src/math/` | `docs/architecture.md` §3.1 (Phase 1) |
| `src/calibration/` | `docs/architecture.md` §3.1 + §8 (data assets) |
| `src/clients/` | `docs/architecture.md` §4 (external services — verified endpoints) |
| `src/engine/` | `docs/architecture.md` §3.3 (Phase 3 — EKF, HMM/DomIndex, MC pipeline) |
| `src/execution/` | `docs/architecture.md` §3.4 + §3.7 (Phase 4 + trading logic) |
| `src/orchestrator/` | `docs/architecture.md` §3.5 (orchestrator) |
| `src/dashboard/` | `docs/architecture.md` §3.6 (dashboard) |
| `src/recorder/` | `docs/architecture.md` Sprint 3 (recording infrastructure) |
| DB schema | `docs/architecture.md` §5 (PostgreSQL + Redis) |
| Docker | `docs/architecture.md` §6 (infrastructure) |

## Project Structure

```
FKT_v4/
├── CLAUDE.md                    ← you are here
├── .claude/rules/
│   ├── coding.md                ← Python conventions
│   └── patterns.md              ← system patterns (read every session)
├── docs/
│   └── architecture.md          ← single source of truth (1,274 lines)
├── src/
│   ├── math/                    ← 4 core files (copied from v3, extend for v5 — preserve old signatures as wrappers)
│   │   ├── mc_core.py           ← Numba JIT MC
│   │   ├── step_1_4_nll_optimize.py  ← Adam→L-BFGS NLL
│   │   ├── step_1_2_Q_estimation.py  ← Q matrix
│   │   └── compute_mu.py       ← remaining μ
│   ├── calibration/             ← Phase 1 pipeline
│   ├── clients/                 ← Goalserve, Kalshi, OddsAPI clients
│   ├── engine/                  ← Phase 3: tick loop, EKF, HMM/DomIndex, events
│   ├── execution/               ← Phase 4: signals, Kelly, exits, settlement
│   ├── orchestrator/            ← scheduler, container lifecycle
│   ├── recorder/                ← live data recording + ReplayServer
│   └── dashboard/               ← FastAPI API + React UI
├── data/
│   ├── commentaries/            ← Goalserve historical (12,607 matches)
│   ├── odds_historical/         ← football-data.co.uk CSVs
│   └── recordings/              ← Sprint 3+ recorded match data
├── keys/                        ← Kalshi RSA key
├── tests/
├── docker/
└── .env                         ← API keys (GOALSERVE, KALSHI, ODDS_API)
```

## Commands

```bash
make test                         # run all tests
make lint                         # ruff + mypy
docker compose up -d              # postgres + redis
docker compose up                 # full stack
```

## Current Progress

- [x] Sprint -1: Feasibility study
- [x] Sprint 0: Project skeleton
- [x] Sprint S1: v5 math core migration (mc_core_v5, asymmetric delta, stoppage eta, sigma_omega)
- [x] Sprint S2: v5 clients + Phase 2 migration (Kalshi WS, Shin vig, ekf_P0)
- [x] Sprint S3: v5 Phase 3 engine migration (EKF, DomIndex, HMM stub, tick_loop v5 pipeline)
- [ ] Sprint 1-7: Implementation