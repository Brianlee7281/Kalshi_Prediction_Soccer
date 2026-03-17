"""LiveMatchModel — shared mutable state for a single live match.

Central state object read/written by all three Phase 3 coroutines
(tick_loop, odds_api_listener, goalserve_poller). Single-threaded
asyncio means no locks are needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from src.common.logging import get_logger
from src.common.types import OddsConsensusResult, Phase2Result

logger = get_logger("engine.model")

# Alias for compute_mu TYPE_CHECKING import
LiveFootballQuantModel = "LiveMatchModel"

# Grid constants
_STANDARD_GRID_MAX = 100  # 1-minute increments, 0..100
_FINE_GRID_MAX = 30  # 10-second increments (0..30 = 0..5 minutes)


def _precompute_grids(
    Q: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Precompute transition probability matrices P(dt) = expm(Q * dt).

    Returns:
        (P_grid, P_fine_grid) where:
        - P_grid[k] = expm(Q * k) for k in 0.._STANDARD_GRID_MAX (minutes)
        - P_fine_grid[k] = expm(Q * k/6) for k in 0.._FINE_GRID_MAX (10-sec steps)
    """
    P_grid: dict[int, np.ndarray] = {}
    for k in range(_STANDARD_GRID_MAX + 1):
        P_grid[k] = expm(Q * float(k))

    P_fine_grid: dict[int, np.ndarray] = {}
    for k in range(_FINE_GRID_MAX + 1):
        dt_min = k / 6.0  # 10-second steps in minutes
        P_fine_grid[k] = expm(Q * dt_min)

    return P_grid, P_fine_grid


@dataclass
class LiveMatchModel:
    """Mutable state for a single live match."""

    # Identity
    match_id: str
    league_id: int
    home_team: str
    away_team: str

    # Phase 2 inputs (immutable after init)
    a_H: float
    a_A: float
    param_version: int
    b: np.ndarray  # shape (8,) time profile
    gamma_H: np.ndarray  # shape (4,) home red card penalties
    gamma_A: np.ndarray  # shape (4,) away red card penalties
    delta_H: np.ndarray  # shape (5,) home score-diff effects
    delta_A: np.ndarray  # shape (5,) away score-diff effects
    Q: np.ndarray  # shape (4,4) generator matrix
    basis_bounds: np.ndarray  # shape (9,) basis period boundaries
    kalshi_tickers: dict[str, str]  # {"home_win": "KX...", ...}

    # Time management (Pattern 3)
    kickoff_wall_clock: float = 0.0
    halftime_start: float = 0.0
    halftime_accumulated: float = 0.0
    t: float = 0.0  # EFFECTIVE play time in minutes
    T_exp: float = 93.0

    # Match state
    engine_phase: str = "WAITING_FOR_KICKOFF"
    score: tuple[int, int] = (0, 0)
    current_state_X: int = 0  # Markov state {0,1,2,3}
    delta_S: int = 0  # score diff (home - away)
    mu_H: float = 0.0
    mu_A: float = 0.0

    # Event tracking
    event_state: str = "IDLE"  # IDLE | PRELIMINARY | CONFIRMED
    cooldown: bool = False
    cooldown_until_tick: int = 0
    ob_freeze: bool = False
    _last_period: str = ""
    _last_score: tuple[int, int] = (0, 0)

    # Tick management
    tick_count: int = 0

    # Odds consensus state (updated by odds_api_listener)
    odds_consensus: OddsConsensusResult | None = None

    # Precomputed grids (for compute_mu)
    P_grid: dict[int, np.ndarray] = field(default_factory=dict)
    P_fine_grid: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_phase2_result(
        cls, result: Phase2Result, params: dict
    ) -> LiveMatchModel:
        """Initialize from Phase2Result + production_params dict.

        Args:
            result: Phase 2 backsolve output.
            params: Raw production_params dict with Q, b, gamma_H/A, delta_H/A.
        """
        Q = np.array(params["Q"], dtype=np.float64)
        b = np.array(params["b"], dtype=np.float64)
        gamma_H = np.array(params["gamma_H"], dtype=np.float64)
        gamma_A = np.array(params["gamma_A"], dtype=np.float64)
        delta_H = np.array(params["delta_H"], dtype=np.float64)
        delta_A = np.array(params["delta_A"], dtype=np.float64)

        T_exp = 93.0
        # basis_bounds: [0, 15, 30, 45+α₁, 60+α₁, 75+α₁, 85+α₁, 90+α₁, T_exp]
        alpha_1 = params.get("alpha_1", 0.0)
        basis_bounds = np.array(
            [
                0.0, 15.0, 30.0,
                45.0 + alpha_1, 60.0 + alpha_1, 75.0 + alpha_1,
                85.0 + alpha_1, 90.0 + alpha_1, T_exp,
            ],
            dtype=np.float64,
        )

        P_grid, P_fine_grid = _precompute_grids(Q)

        model = cls(
            match_id=result.match_id,
            league_id=result.league_id,
            home_team=result.home_team,
            away_team=result.away_team,
            a_H=result.a_H,
            a_A=result.a_A,
            param_version=result.param_version,
            b=b,
            gamma_H=gamma_H,
            gamma_A=gamma_A,
            delta_H=delta_H,
            delta_A=delta_A,
            Q=Q,
            basis_bounds=basis_bounds,
            kalshi_tickers=result.kalshi_tickers,
            T_exp=T_exp,
            P_grid=P_grid,
            P_fine_grid=P_fine_grid,
        )

        logger.info(
            "model_created",
            match_id=result.match_id,
            a_H=result.a_H,
            a_A=result.a_A,
            param_version=result.param_version,
            P_grid_size=len(P_grid),
            P_fine_grid_size=len(P_fine_grid),
        )

        return model

    def update_time(self) -> None:
        """Update model.t from wall clock (Pattern 3).

        t = (monotonic() - kickoff_wall_clock - halftime_accumulated) / 60

        Only updates during active play (FIRST_HALF or SECOND_HALF).
        """
        if self.engine_phase not in ("FIRST_HALF", "SECOND_HALF"):
            return

        now = time.monotonic()
        elapsed_s = now - self.kickoff_wall_clock - self.halftime_accumulated
        self.t = max(0.0, elapsed_s / 60.0)

    def update_T_exp(self, inj_minute: int) -> None:
        """Update T_exp and basis_bounds[-1] when stoppage time is announced.

        Args:
            inj_minute: Announced stoppage time in minutes (e.g., 5 → T_exp=95).
        """
        new_T = 90.0 + inj_minute
        if new_T <= self.T_exp:
            return
        self.T_exp = new_T
        self.basis_bounds[-1] = new_T
        logger.info(
            "T_exp_updated",
            match_id=self.match_id,
            inj_minute=inj_minute,
            new_T_exp=new_T,
        )

    @property
    def order_allowed(self) -> bool:
        """Phase 3 decides 'can trade?', Phase 4 decides 'should trade?'."""
        return (
            not self.cooldown
            and not self.ob_freeze
            and self.event_state == "IDLE"
        )
