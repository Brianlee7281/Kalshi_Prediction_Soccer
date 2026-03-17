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
