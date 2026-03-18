def test_mc_core_import():
    from src.math.mc_core import mc_simulate_remaining
    assert callable(mc_simulate_remaining)


def test_q_estimation_import():
    from src.math.step_1_2_Q_estimation import estimate_Q_global
    assert callable(estimate_Q_global)


def test_nll_import():
    from src.math.step_1_4_nll_optimize import optimize_nll, MMPPModel
    assert callable(optimize_nll)


def test_compute_mu_import():
    from src.math.compute_mu import compute_remaining_mu
    assert callable(compute_remaining_mu)


def test_compute_mu_v5_import():
    from src.math.compute_mu import compute_remaining_mu_v5
    assert callable(compute_remaining_mu_v5)


def test_mc_simulate_v5_importable():
    from src.math.mc_core import mc_simulate_remaining_v5
    assert callable(mc_simulate_remaining_v5)


def test_mc_v5_symmetric_matches_old():
    """v5 with symmetric deltas and zero eta must produce identical results to old wrapper."""
    import numpy as np
    from src.math.mc_core import mc_simulate_remaining, mc_simulate_remaining_v5

    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gamma_H = np.array([0.0, -0.05, 0.05, 0.0])
    gamma_A = np.array([0.0, 0.05, -0.05, 0.0])
    delta_H = np.array([0.1, 0.05, 0.0, -0.05, -0.1])
    delta_A = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    Q_diag = np.array([-0.01, -0.01, -0.01, -0.01])
    Q_off = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
    ])
    basis_bounds = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

    seed = 12345
    N = 500

    old = mc_simulate_remaining(
        30.0, 90.0, 0, 0, 0, 0, -0.3, -0.4,
        b, gamma_H, gamma_A, delta_H, delta_A,
        Q_diag, Q_off, basis_bounds, N, seed,
    )

    new = mc_simulate_remaining_v5(
        30.0, 90.0, 0, 0, 0, 0, -0.3, -0.4,
        b, gamma_H, gamma_A,
        delta_H, delta_H, delta_A, delta_A,  # symmetric: pos=neg
        Q_diag, Q_off, basis_bounds, N, seed,
        0.0, 0.0, 0.0, 0.0,  # no eta
        45.0, 90.0,  # default stoppage starts
    )

    assert np.array_equal(old, new), f"Old and v5 results differ:\nold={old[:5]}\nnew={new[:5]}"


def test_mc_v5_eta_increases_goals():
    """With positive eta in stoppage window, mean goals should increase."""
    import numpy as np
    from src.math.mc_core import mc_simulate_remaining_v5

    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gamma_H = np.zeros(4)
    gamma_A = np.zeros(4)
    delta = np.zeros(5)
    Q_diag = np.array([-0.001, -0.001, -0.001, -0.001])
    Q_off = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
    ])
    # Simulation runs entirely within 2nd half stoppage window (90-100)
    basis_bounds = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 100.0])
    N = 10000

    # No eta — baseline
    no_eta = mc_simulate_remaining_v5(
        90.0, 100.0, 0, 0, 0, 0, -0.5, -0.5,
        b, gamma_H, gamma_A,
        delta, delta, delta, delta,
        Q_diag, Q_off, basis_bounds, N, 999,
        0.0, 0.0, 0.0, 0.0,
        45.0, 90.0,
    )
    mean_no_eta = no_eta[:, 0].mean() + no_eta[:, 1].mean()

    # With eta = 1.0 for 2nd half stoppage
    with_eta = mc_simulate_remaining_v5(
        90.0, 100.0, 0, 0, 0, 0, -0.5, -0.5,
        b, gamma_H, gamma_A,
        delta, delta, delta, delta,
        Q_diag, Q_off, basis_bounds, N, 999,
        0.0, 0.0, 1.0, 1.0,  # eta_H2=1.0, eta_A2=1.0
        45.0, 90.0,
    )
    mean_with_eta = with_eta[:, 0].mean() + with_eta[:, 1].mean()

    assert mean_with_eta > mean_no_eta, (
        f"Expected more goals with eta>0: {mean_with_eta:.2f} vs {mean_no_eta:.2f}"
    )
