import pytest
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
