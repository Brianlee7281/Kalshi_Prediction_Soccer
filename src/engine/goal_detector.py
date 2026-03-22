"""3-layer goal detection and phantom edge suppression.

Layer 1: Detect Kalshi price spikes and suppress normal entries.
Layer 2: Match spikes against precomputed MC fingerprints for adjacent scores.
Layer 3: Wait for confirmation, then trade on inferred P_model at reduced Kelly.

Layers 2+3 require a live model with MC infrastructure.  In replay mode
(no live MC), only Layer 1 activates — spike suppression from p_kalshi
deltas alone.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from src.common.types import MarketProbs

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

log = structlog.get_logger("goal_detector")

# ── Configuration ────────────────────────────────────────────────
SPIKE_THRESHOLD: float = 0.10       # single-tick or windowed delta to trigger Layer 1
FINGERPRINT_INTERVAL: float = 30.0  # seconds between fingerprint recomputation
MATCH_SCORE_THRESHOLD: float = 0.70 # cosine * magnitude threshold for match
CONFIRMATION_TICKS: int = 4         # ticks to wait before acting on match
SPIKE_TIMEOUT: float = 120.0        # seconds before spike suppression expires
INFERRED_KELLY_MULT: float = 0.50   # Kelly multiplier for inferred events
_SPIKE_WINDOW: int = 3              # ticks for rolling delta window


@dataclass
class GoalDetectionResult:
    """Result from GoalDetector.process_tick()."""

    suppress_entries: bool = False
    inferred_score: tuple[int, int] | None = None
    inferred_P_model: MarketProbs | None = None
    kelly_multiplier: float = 1.0
    inference_mismatch: bool = False  # True when official event contradicts


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _combined_match_score(observed: list[float], fingerprint: list[float]) -> float:
    """Cosine similarity * magnitude agreement."""
    norm_obs = _norm(observed)
    norm_fp = _norm(fingerprint)
    if norm_obs < 1e-9 or norm_fp < 1e-9:
        return 0.0
    cosine = _dot(observed, fingerprint) / (norm_obs * norm_fp)
    mag_ratio = min(norm_obs, norm_fp) / max(norm_obs, norm_fp)
    return cosine * mag_ratio


class GoalDetector:
    """Three-layer goal detection integrated into the Phase 3 tick loop.

    In live mode: pass a LiveMatchModel with MC params for Layers 2+3.
    In replay mode: call load_replay_fingerprints() with pre-computed
    fingerprints derived from the recording's actual goal transitions.
    """

    def __init__(self, model: LiveMatchModel | None = None) -> None:
        self._model = model
        self._has_mc = model is not None and hasattr(model, "b")

        # Layer 1 — spike detection
        self._prev_kalshi: dict[str, float] = {}
        self._price_history: list[dict[str, float]] = []
        self._spike_detected_at: float | None = None
        self._spike_start_tick: int = 0

        # Layer 2 — fingerprint table
        self._fingerprints: dict[str, list[float]] = {}
        self._fingerprint_P_models: dict[str, MarketProbs] = {}
        self._current_P_model: MarketProbs | None = None
        self._last_fp_time: float = 0.0
        self._last_fp_score: tuple[int, int] = (-1, -1)
        self._has_replay_fp: bool = False
        self._replay_fingerprints: dict[tuple[int, int], dict] = {}

        # Layer 3 — confirmation
        self._accumulated_delta: list[float] = [0.0, 0.0, 0.0]
        self._tentative_match: str | None = None
        self._tentative_confirmed: bool = False
        self._inferred_score: tuple[int, int] | None = None

        # Event confirmation
        self._inference_mismatch: bool = False

    # ── Replay fingerprint loading ─────────────────────────────

    def load_replay_fingerprints(
        self,
        fingerprint_table: dict[tuple[int, int], dict[str, tuple[list[float], MarketProbs]]],
    ) -> None:
        """Load pre-computed fingerprints for replay testing.

        Args:
            fingerprint_table: {(S_H, S_A): {"home_goal": ([hw,dr,aw], MarketProbs),
                                              "away_goal": ([hw,dr,aw], MarketProbs)}}

        Built from the recording's actual goal transitions — the P_model
        before and after each goal gives the exact fingerprint.
        """
        self._replay_fingerprints = fingerprint_table
        self._has_replay_fp = True
        log.info(
            "replay_fingerprints_loaded",
            score_states=len(fingerprint_table),
        )

    def _apply_replay_fingerprints(self, score: tuple[int, int]) -> None:
        """Switch fingerprints to match the current score from replay table."""
        if not self._has_replay_fp:
            return
        fp_data = self._replay_fingerprints.get(score)
        if fp_data is None:
            return
        for scenario in ("home_goal", "away_goal"):
            if scenario in fp_data:
                vec, p_model = fp_data[scenario]
                self._fingerprints[scenario] = vec
                self._fingerprint_P_models[scenario] = p_model

    # ── Layer 2: Fingerprint computation ─────────────────────────

    async def update_fingerprints(
        self,
        current_P_model: MarketProbs | None = None,
        current_score: tuple[int, int] | None = None,
    ) -> None:
        """Recompute fingerprints for adjacent scores if needed.

        Call every tick.  Only recomputes when FINGERPRINT_INTERVAL has
        elapsed or the score has changed.  Uses replay fingerprints if loaded,
        otherwise requires live MC.
        """
        if current_P_model is not None:
            self._current_P_model = current_P_model

        # Replay mode: load fingerprints from pre-computed table
        if self._has_replay_fp:
            score = current_score
            if score is None and self._model is not None:
                score = self._model.score
            if score is not None and score != self._last_fp_score:
                self._apply_replay_fingerprints(score)
                self._last_fp_score = score
            return

        if not self._has_mc or self._model is None:
            return

        now = time.monotonic()
        score = self._model.score

        if (
            now - self._last_fp_time < FINGERPRINT_INTERVAL
            and score == self._last_fp_score
        ):
            return

        # Lazy import to avoid circular deps
        from src.engine.mc_pricing import compute_mc_for_score

        S_H, S_A = score

        try:
            p_home_goal = await compute_mc_for_score(self._model, S_H + 1, S_A)
            p_away_goal = await compute_mc_for_score(self._model, S_H, S_A + 1)
        except Exception as exc:
            log.warning("fingerprint_mc_error", error=str(exc))
            return

        # Current P_model (already computed this tick)
        if self._current_P_model is None:
            return

        cur = self._current_P_model
        self._fingerprints["home_goal"] = [
            p_home_goal.home_win - cur.home_win,
            p_home_goal.draw - cur.draw,
            p_home_goal.away_win - cur.away_win,
        ]
        self._fingerprints["away_goal"] = [
            p_away_goal.home_win - cur.home_win,
            p_away_goal.draw - cur.draw,
            p_away_goal.away_win - cur.away_win,
        ]
        self._fingerprint_P_models["home_goal"] = p_home_goal
        self._fingerprint_P_models["away_goal"] = p_away_goal

        self._last_fp_time = now
        self._last_fp_score = score

        log.info(
            "fingerprints_updated",
            score=score,
            fp_home_mag=round(_norm(self._fingerprints["home_goal"]), 4),
            fp_away_mag=round(_norm(self._fingerprints["away_goal"]), 4),
        )

    # ── Main tick processing ─────────────────────────────────────

    def process_tick(
        self,
        p_kalshi: dict[str, float],
        tick_count: int,
    ) -> GoalDetectionResult:
        """Process one tick of Kalshi prices through all three layers.

        Args:
            p_kalshi: Current Kalshi mid prices {market_type: float}.
            tick_count: Current tick number.

        Returns:
            GoalDetectionResult with suppression/inference state.
        """
        # Reset mismatch flag each tick (consumed by caller)
        mismatch = self._inference_mismatch
        self._inference_mismatch = False

        # ── Layer 1: Spike detection ─────────────────────────
        if self._prev_kalshi:
            delta_hw = p_kalshi.get("home_win", 0) - self._prev_kalshi.get("home_win", 0)
            delta_dr = p_kalshi.get("draw", 0) - self._prev_kalshi.get("draw", 0)
            delta_aw = p_kalshi.get("away_win", 0) - self._prev_kalshi.get("away_win", 0)
        else:
            delta_hw = delta_dr = delta_aw = 0.0

        # Track price history for windowed spike detection
        self._price_history.append(dict(p_kalshi))
        if len(self._price_history) > _SPIKE_WINDOW + 1:
            self._price_history.pop(0)

        self._prev_kalshi = dict(p_kalshi)

        # Check both single-tick AND windowed delta
        max_single = max(abs(delta_hw), abs(delta_dr), abs(delta_aw))
        # Windowed: compare current price vs price N ticks ago
        max_windowed = 0.0
        win_hw = win_dr = win_aw = 0.0
        if len(self._price_history) > _SPIKE_WINDOW:
            old = self._price_history[0]
            win_hw = p_kalshi.get("home_win", 0) - old.get("home_win", 0)
            win_dr = p_kalshi.get("draw", 0) - old.get("draw", 0)
            win_aw = p_kalshi.get("away_win", 0) - old.get("away_win", 0)
            max_windowed = max(abs(win_hw), abs(win_dr), abs(win_aw))

        spike_triggered = max_single > SPIKE_THRESHOLD or max_windowed > SPIKE_THRESHOLD

        if spike_triggered:
            if self._spike_detected_at is None:
                # New spike — seed with windowed delta (captures multi-tick moves)
                self._spike_detected_at = time.monotonic()
                self._spike_start_tick = tick_count
                self._accumulated_delta = [win_hw, win_dr, win_aw]
                log.info(
                    "spike_detected",
                    tick=tick_count,
                    hw=round(win_hw, 4),
                    dr=round(win_dr, 4),
                    aw=round(win_aw, 4),
                )
            else:
                # Ongoing spike — accumulate
                self._accumulated_delta[0] += delta_hw
                self._accumulated_delta[1] += delta_dr
                self._accumulated_delta[2] += delta_aw
        elif self._spike_detected_at is not None:
            # Below threshold but spike active — still accumulate drift
            self._accumulated_delta[0] += delta_hw
            self._accumulated_delta[1] += delta_dr
            self._accumulated_delta[2] += delta_aw

        # Spike timeout — use both wall clock and tick count
        # (tick count is needed for replay where wall clock doesn't advance)
        if self._spike_detected_at is not None:
            wall_elapsed = time.monotonic() - self._spike_detected_at
            tick_elapsed = tick_count - self._spike_start_tick
            if wall_elapsed > SPIKE_TIMEOUT or tick_elapsed > int(SPIKE_TIMEOUT):
                log.info("spike_timeout", tick=tick_count, elapsed_ticks=tick_elapsed)
                self._clear_spike()

        # ── Layer 2: Fingerprint matching (only with MC) ─────
        if (
            self._spike_detected_at is not None
            and (self._has_mc or self._has_replay_fp)
            and not self._tentative_confirmed
            and self._tentative_match is None
        ):
            ticks_since = tick_count - self._spike_start_tick
            if ticks_since >= CONFIRMATION_TICKS and self._fingerprints:
                best_scenario = None
                best_score = -1.0
                for scenario in ("home_goal", "away_goal"):
                    fp = self._fingerprints.get(scenario)
                    if fp is None:
                        continue
                    score = _combined_match_score(self._accumulated_delta, fp)
                    if score > best_score:
                        best_score = score
                        best_scenario = scenario

                if best_score > MATCH_SCORE_THRESHOLD and best_scenario is not None:
                    self._tentative_match = best_scenario
                    log.info(
                        "fingerprint_matched",
                        scenario=best_scenario,
                        match_score=round(best_score, 4),
                        tick=tick_count,
                    )
                elif ticks_since == CONFIRMATION_TICKS:
                    # Log only once (at the confirmation tick, not every subsequent tick)
                    log.debug(
                        "no_fingerprint_match",
                        best_score=round(best_score, 4),
                        tick=tick_count,
                    )

        # ── Layer 3: Confirmation + inference ────────────────
        if (
            self._tentative_match is not None
            and not self._tentative_confirmed
        ):
            fp = self._fingerprints.get(self._tentative_match)
            if fp is not None:
                recheck = _combined_match_score(self._accumulated_delta, fp)
                if recheck > MATCH_SCORE_THRESHOLD:
                    self._tentative_confirmed = True
                    # Get current score from model or last known state
                    if self._model is not None:
                        S_H, S_A = self._model.score
                    else:
                        S_H, S_A = self._last_fp_score if self._last_fp_score != (-1, -1) else (0, 0)
                    if self._tentative_match == "home_goal":
                        self._inferred_score = (S_H + 1, S_A)
                    else:
                        self._inferred_score = (S_H, S_A + 1)
                    log.info(
                        "inference_confirmed",
                        inferred_score=self._inferred_score,
                        tick=tick_count,
                    )

        # ── Build result ─────────────────────────────────────
        suppress = self._spike_detected_at is not None

        inferred_P = None
        inferred_sc = None
        kelly_mult = 1.0

        if self._tentative_confirmed:
            inferred_sc = self._inferred_score
            inferred_P = self._fingerprint_P_models.get(self._tentative_match)
            kelly_mult = INFERRED_KELLY_MULT
            suppress = False  # allow trading with inferred model

        return GoalDetectionResult(
            suppress_entries=suppress,
            inferred_score=inferred_sc,
            inferred_P_model=inferred_P,
            kelly_multiplier=kelly_mult,
            inference_mismatch=mismatch,
        )

    # ── Event confirmation ───────────────────────────────────

    def on_event_confirmed(self, event_type: str, new_score: tuple[int, int]) -> None:
        """Called when official Goalserve/Kalshi event notification arrives.

        If the inferred score matches, clears the spike cleanly.
        If it doesn't match, sets inference_mismatch flag so the execution
        loop can close positions opened by the fast-react module.
        """
        if self._inferred_score is not None:
            if self._inferred_score == new_score:
                log.info(
                    "inference_correct",
                    inferred=self._inferred_score,
                    actual=new_score,
                )
            else:
                log.warning(
                    "inference_mismatch",
                    inferred=self._inferred_score,
                    actual=new_score,
                )
                self._inference_mismatch = True

        self._clear_spike()

    def _clear_spike(self) -> None:
        """Reset all spike / inference state."""
        self._spike_detected_at = None
        self._spike_start_tick = 0
        self._accumulated_delta = [0.0, 0.0, 0.0]
        self._tentative_match = None
        self._tentative_confirmed = False
        self._inferred_score = None
