"""Phase 4 position tracking and 6-trigger exit logic.

In-memory state management — no database, no network, no API calls.
"""

from __future__ import annotations

import uuid

import structlog

from src.common.types import (
    ExitDecision,
    ExitTrigger,
    FillResult,
    Position,
    Signal,
    TickPayload,
)
from src.execution.config import CONFIG
from src.execution.kelly_sizer import compute_kelly_fraction
from src.execution.signal_generator import (
    _get_market_ekf_P,
    _get_market_mu,
    compute_dynamic_threshold,
)

log = structlog.get_logger("position_monitor")


class PositionTracker:
    """In-memory position state manager with 6-trigger exit evaluation."""

    def __init__(
        self,
        min_hold_ticks: int = CONFIG.MIN_HOLD_TICKS,
        cooldown_after_exit: int = CONFIG.COOLDOWN_AFTER_EXIT,
    ) -> None:
        self.open_positions: dict[str, Position] = {}
        self.exit_cooldowns: dict[str, int] = {}  # market_type → tick when cooldown expires
        self.min_hold_ticks = min_hold_ticks
        self.cooldown_after_exit = cooldown_after_exit

    def add_position(
        self, signal: Signal, fill: FillResult, tick: int, t: float
    ) -> Position:
        """Create and store a new position from a filled signal."""
        if signal.direction == "BUY_YES":
            entry_price = fill.price
        else:
            entry_price = 1.0 - fill.price

        position = Position(
            id=str(uuid.uuid4()),
            match_id=signal.match_id,
            ticker=signal.ticker,
            market_type=signal.market_type,
            direction=signal.direction,
            quantity=fill.quantity,
            entry_price=entry_price,
            entry_tick=tick,
            entry_t=t,
            is_paper=(fill.status == "paper"),
        )
        self.open_positions[position.id] = position
        log.info(
            "position_opened",
            position_id=position.id,
            ticker=signal.ticker,
            direction=signal.direction,
            quantity=fill.quantity,
            entry_price=entry_price,
        )
        return position

    def check_exits(
        self, payload: TickPayload, p_kalshi: dict[str, float]
    ) -> list[ExitDecision]:
        """Evaluate all 6 exit triggers for every open position."""
        decisions: list[ExitDecision] = []

        for position in list(self.open_positions.values()):
            if position.market_type not in p_kalshi:
                continue

            # Update tracking fields
            position.ticks_held += 1
            p_model = getattr(payload.P_model, position.market_type)
            if p_model is None:
                continue
            position.current_p_model = p_model

            p_k = p_kalshi[position.market_type]
            position.current_p_kalshi = p_k

            # Compute current EV based on direction
            if position.direction == "BUY_YES":
                ev = p_model - p_k
            else:
                ev = p_k - p_model

            # Compute shared threshold values
            mu_market = _get_market_mu(
                position.market_type, payload.mu_H, payload.mu_A
            )
            ekf_P = _get_market_ekf_P(
                position.market_type, payload.ekf_P_H, payload.ekf_P_A
            )
            sigma_mc_val = getattr(payload.sigma_MC, position.market_type)
            if sigma_mc_val is None:
                sigma_mc_val = 0.0
            theta = compute_dynamic_threshold(p_model, sigma_mc_val, ekf_P, mu_market)

            held = position.ticks_held
            decision = None

            # Trigger 2 — EDGE_REVERSAL (ignores min_hold)
            # Symmetric threshold: exit only when the edge has reversed by more
            # than theta (the same dynamic threshold used for entry).  A mere
            # zero-crossing is noise; we need the market to convincingly disagree.
            if position.direction == "BUY_YES" and (p_k - p_model) > theta:
                decision = ExitDecision(
                    position_id=position.id,
                    trigger=ExitTrigger.EDGE_REVERSAL,
                    contracts_to_exit=position.quantity,
                    exit_price=p_k,
                    reason=f"edge_reversal: neg_edge={p_k - p_model:.4f} > theta={theta:.4f}",
                )
            elif position.direction == "BUY_NO" and (p_model - p_k) > theta:
                decision = ExitDecision(
                    position_id=position.id,
                    trigger=ExitTrigger.EDGE_REVERSAL,
                    contracts_to_exit=position.quantity,
                    exit_price=p_k,
                    reason=f"edge_reversal: neg_edge={p_model - p_k:.4f} > theta={theta:.4f}",
                )

            # Trigger 6 — EKF_DIVERGENCE (ignores min_hold, safety)
            if decision is None and (
                payload.ekf_P_H > CONFIG.EKF_DIVERGENCE_THRESHOLD
                or payload.ekf_P_A > CONFIG.EKF_DIVERGENCE_THRESHOLD
            ):
                decision = ExitDecision(
                    position_id=position.id,
                    trigger=ExitTrigger.EKF_DIVERGENCE,
                    contracts_to_exit=position.quantity,
                    exit_price=p_k,
                    reason=f"ekf_divergence: P_H={payload.ekf_P_H:.3f} P_A={payload.ekf_P_A:.3f}",
                )

            # Remaining triggers require min_hold
            if decision is None and held >= self.min_hold_ticks:
                # Trigger 1 — EDGE_DECAY
                if ev < theta:
                    decision = ExitDecision(
                        position_id=position.id,
                        trigger=ExitTrigger.EDGE_DECAY,
                        contracts_to_exit=position.quantity,
                        exit_price=p_k,
                        reason=f"edge_decay: ev={ev:.4f} < theta={theta:.4f}",
                    )

                # Trigger 3 — POSITION_TRIM
                if decision is None and p_k > 0:
                    kelly_frac = compute_kelly_fraction(p_model, p_k)
                    kelly_optimal = max(1, int(kelly_frac * 10000.0 / p_k))
                    if position.quantity > 2 * kelly_optimal:
                        contracts_to_exit = position.quantity - kelly_optimal
                        decision = ExitDecision(
                            position_id=position.id,
                            trigger=ExitTrigger.POSITION_TRIM,
                            contracts_to_exit=contracts_to_exit,
                            exit_price=p_k,
                            reason=f"position_trim: {position.quantity} > 2x optimal {kelly_optimal}",
                        )

                # Trigger 4 — OPPORTUNITY_COST
                if decision is None:
                    if position.direction == "BUY_YES":
                        opposite_ev = p_k - p_model
                    else:
                        opposite_ev = p_model - p_k
                    if opposite_ev > theta:
                        decision = ExitDecision(
                            position_id=position.id,
                            trigger=ExitTrigger.OPPORTUNITY_COST,
                            contracts_to_exit=position.quantity,
                            exit_price=p_k,
                            reason=f"opportunity_cost: opposite_ev={opposite_ev:.4f} > theta={theta:.4f}",
                        )

                # Trigger 5 — EXPIRY_EVAL
                if decision is None and payload.t > CONFIG.EXPIRY_EVAL_MINUTE:
                    if ev < CONFIG.C_SPREAD:
                        decision = ExitDecision(
                            position_id=position.id,
                            trigger=ExitTrigger.EXPIRY_EVAL,
                            contracts_to_exit=position.quantity,
                            exit_price=p_k,
                            reason=f"expiry_eval: t={payload.t:.1f} ev={ev:.4f}",
                        )

            if decision is not None:
                decisions.append(decision)

        return decisions

    def close_position(
        self,
        position_id: str,
        exit_trigger: ExitTrigger,
        contracts_exited: int,
        exit_price: float,
        current_tick: int,
    ) -> Position:
        """Close (fully or partially) a position."""
        position = self.open_positions[position_id]

        if contracts_exited >= position.quantity:
            # Full exit
            del self.open_positions[position_id]
            self.exit_cooldowns[position.market_type] = (
                current_tick + self.cooldown_after_exit
            )
            log.info(
                "position_closed",
                position_id=position_id,
                trigger=exit_trigger.value,
                contracts=contracts_exited,
            )
        else:
            # Partial exit (POSITION_TRIM)
            position.quantity -= contracts_exited
            log.info(
                "position_trimmed",
                position_id=position_id,
                contracts_exited=contracts_exited,
                remaining=position.quantity,
            )

        return position

    def is_in_cooldown(self, market_type: str, current_tick: int) -> bool:
        """Check if a market is in post-exit cooldown."""
        return current_tick < self.exit_cooldowns.get(market_type, 0)

    def get_total_exposure(self) -> float:
        """Total dollar exposure across all open positions."""
        return sum(
            pos.quantity * pos.entry_price for pos in self.open_positions.values()
        )

    def get_match_exposure(self, match_id: str) -> float:
        """Dollar exposure for a specific match."""
        return sum(
            pos.quantity * pos.entry_price
            for pos in self.open_positions.values()
            if pos.match_id == match_id
        )
