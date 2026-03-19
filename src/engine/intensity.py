"""Shared intensity computation — used by tick_loop and event_handlers.

Extracted from tick_loop.py so that handle_goal can compute lambda_H/lambda_A
for the v5 EKF goal-update path without creating a circular import.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel


def basis_index(t: float, basis_bounds: object) -> int:
    """Find which basis period *t* falls into."""
    for i in range(len(basis_bounds) - 1):
        if t < float(basis_bounds[i + 1]):
            return i
    return len(basis_bounds) - 2


def compute_lambda(model: LiveMatchModel, team: str) -> float:
    """Current goal intensity lambda for *team* at ``model.t``.

    lambda = exp(a + b[basis_index] + gamma[state])
    """
    bi = basis_index(model.t, model.basis_bounds)
    b_val = float(model.b[bi]) if bi < len(model.b) else 0.0
    st = model.current_state_X
    if team == "home":
        return math.exp(model.a_H + b_val + float(model.gamma_H[st]))
    return math.exp(model.a_A + b_val + float(model.gamma_A[st]))
