-- These tables were defined in architecture.md §5.1 but may not be created yet.
-- This migration creates them if they don't exist.

CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INT NOT NULL,
    entry_price DECIMAL(6,4) NOT NULL,
    exit_price DECIMAL(6,4),
    status TEXT NOT NULL DEFAULT 'OPEN',
    is_paper BOOLEAN NOT NULL DEFAULT TRUE,
    realized_pnl DECIMAL(10,2),
    entry_tick INT,
    exit_tick INT,
    entry_reason TEXT,
    exit_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_positions_match ON positions(match_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

CREATE TABLE IF NOT EXISTS exposure_reservation (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    reserved_amount DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'RESERVED',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_reservation_status ON exposure_reservation(status);

CREATE TABLE IF NOT EXISTS bankroll (
    mode TEXT PRIMARY KEY,
    balance DECIMAL(12,2) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bankroll_snapshot (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    balance DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO bankroll (mode, balance) VALUES ('paper', 10000.00) ON CONFLICT (mode) DO NOTHING;
INSERT INTO bankroll (mode, balance) VALUES ('live', 0.00) ON CONFLICT (mode) DO NOTHING;
