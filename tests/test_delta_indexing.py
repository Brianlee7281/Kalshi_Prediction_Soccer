"""Tests for delta indexing: away team must use away perspective (-ds + 2).

The MLE calibration (phase1_mle.py) indexes delta arrays from each team's
own perspective:
  di_H = clamp(ds + 2, 0, 4)   — home perspective
  di_A = clamp(-ds + 2, 0, 4)  — away perspective

The engine (compute_mu, mc_core) must match this convention.
"""

import numpy as np
import pytest

from src.math.mc_core import mc_simulate_remaining, mc_simulate_remaining_v5


# ── Helpers ──────────────────────────────────────────────────────────

def _make_simple_params():
    """Create minimal params for MC testing."""
    return {
        "b": np.zeros(8, dtype=np.float64),
        "gamma_H": np.zeros(4, dtype=np.float64),
        "gamma_A": np.zeros(4, dtype=np.float64),
        "Q_diag": np.array([-0.001, -0.001, -0.001, -0.001], dtype=np.float64),
        "Q_off": np.eye(4, dtype=np.float64) * 0.0,  # no red card transitions
        "basis_bounds": np.array([0, 15, 30, 45, 60, 75, 85, 90, 93], dtype=np.float64),
    }


# ── MC delta indexing ────────────────────────────────────────────────

def test_mc_delta_indexing_home_leading():
    """When home leads by 2 (ds=+2), away should get delta_A[0], not delta_A[4].

    Use asymmetric delta_A to detect the indexing:
      delta_A = [-0.5, -0.25, 0, +0.25, +0.5]
    Correct (di_A=0): away rate *= exp(-0.5) = 0.61 → fewer away goals → low P(draw)
    Wrong   (di_A=4): away rate *= exp(+0.5) = 1.65 → more away goals → high P(draw)
    """
    p = _make_simple_params()

    # Balanced base rates so delta effect is clearly visible
    a_H = -4.5
    a_A = -4.5
    delta_H = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta_A = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)

    # Score 3-1 (home leading by 2), t=40, plenty of time remaining
    results = mc_simulate_remaining(
        t_now=40.0, T_end=93.0, S_H=3, S_A=1,
        state=0, score_diff=2,
        a_H=a_H, a_A=a_A,
        b=p["b"], gamma_H=p["gamma_H"], gamma_A=p["gamma_A"],
        delta_H=delta_H, delta_A=delta_A,
        Q_diag=p["Q_diag"], Q_off=p["Q_off"],
        basis_bounds=p["basis_bounds"],
        N=50_000, seed=42,
    )
    N = len(results)
    p_draw = float(np.sum(results[:, 0] == results[:, 1])) / N

    # With correct indexing (delta_A[0]=-0.5), away rate is halved.
    # From 3-1, draw requires away to net +2 goals — should be very unlikely (<10%)
    # With wrong indexing (delta_A[4]=+0.5), away rate boosted → P(draw) much higher
    assert p_draw < 0.10, (
        f"P(draw) = {p_draw:.3f} at 3-1 with delta_A[0]=-0.5 should be <10%. "
        f"If >15%, delta_A is likely indexed from home perspective (bug)."
    )


def test_mc_delta_indexing_away_leading():
    """When away leads by 2 (ds=-2), home should get delta_H[0], not delta_H[4].

    Symmetric test to verify home team delta is also correctly indexed.
    """
    p = _make_simple_params()

    a_H = -4.5
    a_A = -4.5
    delta_H = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)
    delta_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Score 1-3 (away leading by 2)
    results = mc_simulate_remaining(
        t_now=40.0, T_end=93.0, S_H=1, S_A=3,
        state=0, score_diff=-2,
        a_H=a_H, a_A=a_A,
        b=p["b"], gamma_H=p["gamma_H"], gamma_A=p["gamma_A"],
        delta_H=delta_H, delta_A=delta_A,
        Q_diag=p["Q_diag"], Q_off=p["Q_off"],
        basis_bounds=p["basis_bounds"],
        N=50_000, seed=42,
    )
    N = len(results)
    p_draw = float(np.sum(results[:, 0] == results[:, 1])) / N

    # delta_H[0]=-0.5 (home trailing by 2, scores less) → hard to catch up
    assert p_draw < 0.10, (
        f"P(draw) = {p_draw:.3f} at 1-3 with delta_H[0]=-0.5 should be <10%."
    )


def test_mc_delta_indexing_tied_score():
    """At tied score (ds=0), both teams should get delta index 2 (reference=0).

    This should be unaffected by the fix since di_H = di_A = 2 when ds=0.
    """
    p = _make_simple_params()

    a_H = -4.5
    a_A = -4.5
    delta_H = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)
    delta_A = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)

    # Score 1-1 (tied)
    results = mc_simulate_remaining(
        t_now=40.0, T_end=93.0, S_H=1, S_A=1,
        state=0, score_diff=0,
        a_H=a_H, a_A=a_A,
        b=p["b"], gamma_H=p["gamma_H"], gamma_A=p["gamma_A"],
        delta_H=delta_H, delta_A=delta_A,
        Q_diag=p["Q_diag"], Q_off=p["Q_off"],
        basis_bounds=p["basis_bounds"],
        N=50_000, seed=42,
    )
    N = len(results)
    p_draw = float(np.sum(results[:, 0] == results[:, 1])) / N

    # At 1-1 with equal strengths and delta=0, P(draw) should be reasonable (30-50%)
    assert 0.20 < p_draw < 0.60, (
        f"P(draw) = {p_draw:.3f} at 1-1 with delta=0 should be 20-60%."
    )


def test_mc_v5_delta_indexing():
    """v5 asymmetric delta MC also uses correct per-team indexing."""
    p = _make_simple_params()

    a_H = -4.5
    a_A = -4.5
    # All asymmetric arrays: use distinct values to detect swaps
    delta_H_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta_H_neg = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta_A_pos = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)
    delta_A_neg = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)

    # Score 3-1 (home leading by 2, ds=+2)
    results = mc_simulate_remaining_v5(
        t_now=40.0, T_end=93.0, S_H=3, S_A=1,
        state=0, score_diff=2,
        a_H=a_H, a_A=a_A,
        b=p["b"], gamma_H=p["gamma_H"], gamma_A=p["gamma_A"],
        delta_H_pos=delta_H_pos, delta_H_neg=delta_H_neg,
        delta_A_pos=delta_A_pos, delta_A_neg=delta_A_neg,
        Q_diag=p["Q_diag"], Q_off=p["Q_off"],
        basis_bounds=p["basis_bounds"],
        N=50_000, seed=42,
        eta_H=0.0, eta_A=0.0, eta_H2=0.0, eta_A2=0.0,
        stoppage_1_start=45.0, stoppage_2_start=90.0,
    )
    N = len(results)
    p_draw = float(np.sum(results[:, 0] == results[:, 1])) / N

    # sd>0 → uses delta_A_pos. Correct: di_A=0 → delta_A_pos[0]=-0.5
    assert p_draw < 0.10, (
        f"v5 P(draw) = {p_draw:.3f} at 3-1 should be <10% with delta_A_pos[0]=-0.5."
    )


def test_delta_index_values():
    """Direct arithmetic check of index computation."""
    # ds=+2 (home up 2): di_H=4, di_A=0
    ds = 2
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))
    assert di_H == 4
    assert di_A == 0

    # ds=-1 (away up 1): di_H=1, di_A=3
    ds = -1
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))
    assert di_H == 1
    assert di_A == 3

    # ds=0 (tied): di_H=2, di_A=2
    ds = 0
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))
    assert di_H == 2
    assert di_A == 2

    # ds=+3 (clamped to +2): di_H=4, di_A=0
    ds = 3
    di_H = max(0, min(4, ds + 2))
    di_A = max(0, min(4, -ds + 2))
    assert di_H == 4
    assert di_A == 0
