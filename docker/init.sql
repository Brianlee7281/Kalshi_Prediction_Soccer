-- Phase 1 trained parameters
CREATE TABLE production_params (
    version SERIAL PRIMARY KEY,
    league_id INT NOT NULL,
    Q JSONB NOT NULL,
    b JSONB NOT NULL,
    gamma_H DECIMAL(8,6) NOT NULL,
    gamma_A DECIMAL(8,6) NOT NULL,
    delta_H DECIMAL(8,6) NOT NULL,
    delta_A DECIMAL(8,6) NOT NULL,
    sigma_a DECIMAL(8,6) NOT NULL,
    xgb_model_blob BYTEA,
    feature_mask JSONB,
    trained_at TIMESTAMPTZ NOT NULL,
    match_count INT NOT NULL,
    brier_score DECIMAL(6,4) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE
);

-- Match lifecycle
CREATE TABLE match_schedule (
    match_id TEXT PRIMARY KEY,
    league_id INT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_utc TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'SCHEDULED',
    -- SCHEDULED → PHASE2_RUNNING → PHASE2_DONE/SKIPPED → PHASE3_RUNNING → FINISHED → ARCHIVED
    trading_mode TEXT NOT NULL DEFAULT 'paper',
    param_version INT,
    kalshi_tickers JSONB,
    goalserve_fix_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Live tick data (Phase 3 output)
CREATE TABLE tick_snapshots (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    t DECIMAL(6,3) NOT NULL,
    engine_phase TEXT NOT NULL,
    p_home_win DECIMAL(6,4),
    p_draw DECIMAL(6,4),
    p_away_win DECIMAL(6,4),
    sigma_home_win DECIMAL(6,4),
    sigma_draw DECIMAL(6,4),
    sigma_away_win DECIMAL(6,4),
    score_home INT,
    score_away INT,
    mu_H DECIMAL(8,4),
    mu_A DECIMAL(8,4),
    order_allowed BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_ticks_match ON tick_snapshots(match_id, t);

-- Events (goals, red cards, period changes)
CREATE TABLE event_log (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    event_type TEXT NOT NULL,
    t DECIMAL(6,3) NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_events_match ON event_log(match_id);

-- Trading positions
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INT NOT NULL,
    entry_price DECIMAL(6,4) NOT NULL,
    exit_price DECIMAL(6,4),
    status TEXT NOT NULL DEFAULT 'OPEN',
    -- OPEN → CLOSED | SETTLED
    is_paper BOOLEAN NOT NULL DEFAULT TRUE,
    realized_pnl DECIMAL(10,2),
    entry_tick INT,
    exit_tick INT,
    entry_reason TEXT,
    exit_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);
CREATE INDEX idx_positions_match ON positions(match_id);
CREATE INDEX idx_positions_status ON positions(status);

-- Cross-container exposure reservation
CREATE TABLE exposure_reservation (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    reserved_amount DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'RESERVED',
    -- RESERVED → CONFIRMED | RELEASED
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX idx_reservation_status ON exposure_reservation(status);

-- Bankroll (paper and live separate)
CREATE TABLE bankroll (
    mode TEXT PRIMARY KEY,              -- 'paper' | 'live'
    balance DECIMAL(12,2) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bankroll history for drawdown tracking
CREATE TABLE bankroll_snapshot (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    balance DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ticker mapping cache
CREATE TABLE ticker_mapping (
    match_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    kalshi_ticker TEXT NOT NULL,
    PRIMARY KEY (match_id, market_type)
);
