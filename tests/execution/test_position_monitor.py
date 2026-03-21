"""Tests for src/execution/position_monitor.py."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.common.types import (
    ExitTrigger,
    FillResult,
    MarketProbs,
    Signal,
    TickPayload,
)
from src.execution.position_monitor import PositionTracker


def _make_signal(
    direction: str = "BUY_YES",
    market_type: str = "home_win",
    p_model: float = 0.62,
    p_kalshi: float = 0.55,
) -> Signal:
    return Signal(
        match_id="test_match",
        ticker="TICKER-HOME",
        market_type=market_type,
        direction=direction,
        P_kalshi=p_kalshi,
        P_model=p_model,
        EV=abs(p_model - p_kalshi),
        kelly_fraction=0.0,
        kelly_amount=0.0,
        contracts=0,
    )


def _make_fill(
    price: float = 0.55,
    quantity: int = 10,
    status: str = "paper",
) -> FillResult:
    return FillResult(
        order_id="order-1",
        ticker="TICKER-HOME",
        direction="BUY_YES",
        quantity=quantity,
        price=price,
        status=status,
        fill_cost=quantity * price,
        timestamp=datetime(2026, 3, 18),
    )


def _make_payload(
    order_allowed: bool = True,
    home_win: float = 0.62,
    draw: float = 0.25,
    away_win: float = 0.25,
    over_25: float = 0.50,
    btts_yes: float = 0.40,
    mu_H: float = 1.0,
    mu_A: float = 0.8,
    ekf_P_H: float = 0.10,
    ekf_P_A: float = 0.10,
    surprise_score: float = 0.0,
    t: float = 45.0,
) -> TickPayload:
    return TickPayload(
        match_id="test_match",
        t=t,
        engine_phase="SECOND_HALF",
        P_model=MarketProbs(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            over_25=over_25,
            btts_yes=btts_yes,
        ),
        sigma_MC=MarketProbs(
            home_win=0.003,
            draw=0.003,
            away_win=0.003,
            over_25=0.003,
            btts_yes=0.003,
        ),
        score=(1, 0),
        X=0,
        delta_S=1,
        mu_H=mu_H,
        mu_A=mu_A,
        a_H_current=0.3,
        a_A_current=-0.1,
        ekf_P_H=ekf_P_H,
        ekf_P_A=ekf_P_A,
        surprise_score=surprise_score,
        order_allowed=order_allowed,
        cooldown=False,
        ob_freeze=False,
        event_state="IDLE",
    )


# ── add_position ──────────────────────────────────────────────


def test_add_position_stored():
    tracker = PositionTracker()
    signal = _make_signal()
    fill = _make_fill()
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)
    assert pos.id in tracker.open_positions
    assert tracker.open_positions[pos.id].quantity == 10


def test_add_position_entry_price_buy_yes():
    tracker = PositionTracker()
    signal = _make_signal(direction="BUY_YES")
    fill = _make_fill(price=0.55)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)
    assert pos.entry_price == 0.55


def test_add_position_entry_price_buy_no():
    tracker = PositionTracker()
    signal = _make_signal(direction="BUY_NO")
    fill = _make_fill(price=0.55)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)
    assert pos.entry_price == pytest.approx(0.45)


# ── min_hold and triggers ────────────────────────────────────


def test_min_hold_respected():
    """No exit before min_hold with positive edge (not reversal/divergence)."""
    tracker = PositionTracker(min_hold_ticks=150)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=10)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    # Feed 100 ticks with positive edge — should NOT trigger any exit
    p_kalshi = {"home_win": 0.55}
    payload = _make_payload(home_win=0.62, ekf_P_H=0.05, mu_H=0.5)
    for _ in range(100):
        decisions = tracker.check_exits(payload, p_kalshi)
        # Only EDGE_REVERSAL and EKF_DIVERGENCE ignore min_hold — neither applies here
        assert len(decisions) == 0


def test_edge_reversal_ignores_min_hold():
    """EDGE_REVERSAL fires immediately even before min_hold.

    Requires negative edge to exceed dynamic theta (symmetric threshold).
    Use low EKF uncertainty + large reversal to guarantee theta < neg_edge.
    """
    tracker = PositionTracker(min_hold_ticks=150)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=10)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)

    # Large reversal with low uncertainty so neg_edge > theta
    p_kalshi = {"home_win": 0.80}
    payload = _make_payload(
        home_win=0.20, ekf_P_H=0.01, mu_H=0.3,
    )  # neg_edge = 0.60, theta ≈ 0.03
    for _ in range(4):
        tracker.check_exits(payload, p_kalshi)

    decisions = tracker.check_exits(payload, p_kalshi)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EDGE_REVERSAL
    assert decisions[0].contracts_to_exit == pos.quantity


def test_edge_decay_exit():
    """EDGE_DECAY fires after min_hold when edge disappears."""
    tracker = PositionTracker(min_hold_ticks=150)
    signal = _make_signal(direction="BUY_YES", p_model=0.70, p_kalshi=0.50)
    fill = _make_fill(price=0.50, quantity=10)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    # Feed 149 ticks with good edge — no exit
    good_payload = _make_payload(home_win=0.70, ekf_P_H=0.05, mu_H=0.5)
    p_kalshi_good = {"home_win": 0.50}
    for _ in range(149):
        decisions = tracker.check_exits(good_payload, p_kalshi_good)
        assert len(decisions) == 0

    # Tick 150+: edge has decayed (small EV below threshold)
    weak_payload = _make_payload(home_win=0.52, ekf_P_H=0.10, mu_H=1.0)
    p_kalshi_weak = {"home_win": 0.51}
    decisions = tracker.check_exits(weak_payload, p_kalshi_weak)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EDGE_DECAY


def test_position_trim_partial():
    """POSITION_TRIM does a partial exit when quantity > 2x kelly optimal."""
    tracker = PositionTracker(min_hold_ticks=5)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=20)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)

    # With p_model=0.62, p_kalshi=0.55:
    # kelly_frac ≈ 0.155, kelly_optimal = max(1, int(0.155 * 10000 / 0.55)) = max(1, int(2818)) = 2818
    # 20 is NOT > 2*2818, so we need extreme values.
    # Use p_model ≈ p_kalshi so kelly_frac is tiny → kelly_optimal is small
    payload = _make_payload(home_win=0.56, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi = {"home_win": 0.55}

    # With p_model=0.56, p_kalshi=0.55:
    # kelly_frac = (b*p - q)/b where b=(1/0.55)-1=0.8182, f*=(0.8182*0.56-0.44)/0.8182 ≈ 0.0222
    # kelly_optimal = max(1, int(0.0222 * 10000 / 0.55)) = max(1, int(403)) = 403
    # 20 is NOT > 2*403. Need even smaller kelly.

    # Use p_model very close to p_kalshi
    payload2 = _make_payload(home_win=0.551, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi2 = {"home_win": 0.55}
    # kelly_frac = (0.8182*0.551 - 0.449)/0.8182 ≈ 0.00182
    # kelly_optimal = max(1, int(0.00182 * 10000 / 0.55)) = max(1, int(33)) = 33
    # Still 20 < 2*33. Need kelly_optimal < 10.

    # kelly_frac needs to be < 0.00055 for kelly_optimal=max(1,...) ≤ 10... tough.
    # Actually with min kelly_optimal=1, we need quantity > 2*1 = 2. So we need kelly_frac = 0.
    # Better: just use min_hold=0 and a scenario where position is oversized.
    # Use a mock scenario: p_model=0.551, p_kalshi=0.55 → tiny kelly → kelly_optimal small
    # Actually let's re-examine: kelly_frac = (0.8182*0.551-0.449)/0.8182
    # = (0.45091 - 0.449)/0.8182 = 0.00191/0.8182 = 0.00233
    # kelly_optimal = max(1, int(0.00233 * 10000 / 0.55)) = max(1, int(42.4)) = 42
    # 20 < 84, no trigger.

    # Let's set p_model = p_kalshi exactly → kelly_frac = 0 → kelly_optimal = 1 → 20 > 2
    payload3 = _make_payload(home_win=0.55, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi3 = {"home_win": 0.55}

    # Burn through min_hold ticks first
    for _ in range(5):
        tracker.check_exits(payload3, p_kalshi3)

    # At this point ticks_held=6 > min_hold=5
    # But wait: p_model=0.55, p_kalshi=0.55 → ev=0, which triggers EDGE_DECAY first.
    # We need edge > threshold but kelly_optimal small.
    # Use large edge but tiny kelly scenario — impossible since kelly IS the edge.

    # Different approach: just set huge quantity and reasonable kelly
    # Reset tracker
    tracker2 = PositionTracker(min_hold_ticks=5)
    signal2 = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill2 = _make_fill(price=0.55, quantity=10000)
    pos2 = tracker2.add_position(signal2, fill2, tick=0, t=10.0)

    payload4 = _make_payload(home_win=0.56, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi4 = {"home_win": 0.55}

    for _ in range(6):
        tracker2.check_exits(payload4, p_kalshi4)

    # kelly_frac ≈ 0.0222, kelly_optimal = max(1, int(0.0222*10000/0.55)) = 403
    # 10000 > 2*403 = 806 → POSITION_TRIM fires
    # BUT ev=0.56-0.55=0.01 which may be < theta → EDGE_DECAY fires first
    # theta = 0.01 + 0.005 + 1.645*sigma_p ≈ 0.015 + small = ~0.016
    # ev=0.01 < 0.016 → EDGE_DECAY fires. Need bigger edge.

    tracker3 = PositionTracker(min_hold_ticks=5)
    signal3 = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill3 = _make_fill(price=0.55, quantity=10000)
    pos3 = tracker3.add_position(signal3, fill3, tick=0, t=10.0)

    # p_model=0.62, p_kalshi=0.55 → ev=0.07, should exceed threshold
    payload5 = _make_payload(home_win=0.62, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi5 = {"home_win": 0.55}

    for _ in range(6):
        decisions = tracker3.check_exits(payload5, p_kalshi5)

    # kelly_frac ≈ 0.1556, kelly_optimal = max(1, int(0.1556*10000/0.55)) = 2828
    # 10000 > 2*2828 = 5656 → POSITION_TRIM fires
    # contracts_to_exit = 10000 - 2828 = 7172
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.POSITION_TRIM
    assert decisions[0].contracts_to_exit == 10000 - 2828

    # Now close with partial
    tracker3.close_position(
        pos3.id,
        ExitTrigger.POSITION_TRIM,
        decisions[0].contracts_to_exit,
        0.55,
        current_tick=7,
    )
    assert pos3.id in tracker3.open_positions
    assert tracker3.open_positions[pos3.id].quantity == 2828


def test_position_trim_not_triggered():
    """No trim when quantity <= 2x kelly optimal."""
    tracker = PositionTracker(min_hold_ticks=5)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=8)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    payload = _make_payload(home_win=0.62, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi = {"home_win": 0.55}

    for _ in range(6):
        decisions = tracker.check_exits(payload, p_kalshi)

    # kelly_optimal = 2818, 8 < 2*2818 → no trim
    # ev=0.07 > theta → no EDGE_DECAY
    # no reversal, no ekf divergence
    trim_decisions = [d for d in decisions if d.trigger == ExitTrigger.POSITION_TRIM]
    assert len(trim_decisions) == 0


def test_ekf_divergence():
    """EKF_DIVERGENCE fires at tick=1, ignoring min_hold."""
    tracker = PositionTracker(min_hold_ticks=150)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=10)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    payload = _make_payload(home_win=0.62, ekf_P_H=2.0, ekf_P_A=0.10)
    p_kalshi = {"home_win": 0.55}

    decisions = tracker.check_exits(payload, p_kalshi)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EKF_DIVERGENCE


# ── cooldown ──────────────────────────────────────────────────


def test_cooldown_after_full_exit():
    tracker = PositionTracker(cooldown_after_exit=300)
    signal = _make_signal()
    fill = _make_fill(quantity=10)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)

    tracker.close_position(
        pos.id, ExitTrigger.EDGE_REVERSAL, 10, 0.55, current_tick=100
    )
    assert tracker.is_in_cooldown("home_win", 101) is True
    assert tracker.is_in_cooldown("home_win", 399) is True
    assert tracker.is_in_cooldown("home_win", 400) is False
    assert tracker.is_in_cooldown("home_win", 401) is False


def test_cooldown_not_set_on_partial_exit():
    tracker = PositionTracker(cooldown_after_exit=300)
    signal = _make_signal()
    fill = _make_fill(quantity=20)
    pos = tracker.add_position(signal, fill, tick=0, t=10.0)

    tracker.close_position(
        pos.id, ExitTrigger.POSITION_TRIM, 5, 0.55, current_tick=100
    )
    assert tracker.is_in_cooldown("home_win", 101) is False


# ── multiple positions ────────────────────────────────────────


def test_multiple_positions_independent():
    """Two positions in different markets; only one triggers exit."""
    tracker = PositionTracker(min_hold_ticks=5)

    sig_home = _make_signal(direction="BUY_YES", market_type="home_win", p_model=0.62, p_kalshi=0.55)
    fill_home = _make_fill(price=0.55, quantity=10)
    tracker.add_position(sig_home, fill_home, tick=0, t=10.0)

    sig_away = _make_signal(direction="BUY_YES", market_type="away_win", p_model=0.40, p_kalshi=0.30)
    fill_away = _make_fill(price=0.30, quantity=10)
    tracker.add_position(sig_away, fill_away, tick=0, t=10.0)

    # home_win: p_model=0.62 > p_kalshi=0.55 → no reversal, positive edge
    # away_win: p_model=0.05 vs p_kalshi=0.60 → large EDGE_REVERSAL
    payload = _make_payload(home_win=0.62, away_win=0.05, ekf_P_A=0.01, mu_A=0.3)
    p_kalshi = {"home_win": 0.55, "away_win": 0.60}

    decisions = tracker.check_exits(payload, p_kalshi)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EDGE_REVERSAL


def test_exits_evaluated_when_order_not_allowed():
    """Exits are always checked even when order_allowed=False."""
    tracker = PositionTracker(min_hold_ticks=150)
    signal = _make_signal(direction="BUY_YES", p_model=0.62, p_kalshi=0.55)
    fill = _make_fill(price=0.55, quantity=10)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    # order_allowed=False, but edge reversed by large amount → still gets exit
    payload = _make_payload(order_allowed=False, home_win=0.10, ekf_P_H=0.01, mu_H=0.3)
    p_kalshi = {"home_win": 0.70}

    decisions = tracker.check_exits(payload, p_kalshi)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EDGE_REVERSAL


# ── production constants integration ─────────────────────────


def test_production_params_no_early_exit():
    """With production MIN_HOLD_TICKS=150, no exit fires before tick 150
    (except EDGE_REVERSAL and EKF_DIVERGENCE which ignore min_hold)."""
    tracker = PositionTracker()  # uses CONFIG defaults: 150, 300
    signal = _make_signal(direction="BUY_YES", p_model=0.70, p_kalshi=0.50)
    fill = _make_fill(price=0.50, quantity=10)
    tracker.add_position(signal, fill, tick=0, t=10.0)

    # Feed 149 ticks with positive edge — no exits
    payload = _make_payload(home_win=0.70, ekf_P_H=0.05, mu_H=0.5)
    p_kalshi = {"home_win": 0.50}
    for _ in range(149):
        decisions = tracker.check_exits(payload, p_kalshi)
        assert len(decisions) == 0

    # At tick 150 with large reversed edge + low uncertainty → EDGE_REVERSAL fires
    reversed_payload = _make_payload(home_win=0.10, ekf_P_H=0.01, mu_H=0.3)
    decisions = tracker.check_exits(reversed_payload, p_kalshi)
    assert len(decisions) == 1
    assert decisions[0].trigger == ExitTrigger.EDGE_REVERSAL
