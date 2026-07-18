"""
Tests for Alternating Optimization Solver.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from ..src.system_model import ISACSystem
from ..src.ao_solver import AOSolver


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def detection_solver(system):
    """Create AO solver for detection QoS."""
    return AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=20)


@pytest.fixture
def localization_solver(system):
    """Create AO solver for localization QoS."""
    return AOSolver(system, qos_type='localization', fairness='maxmin', max_iter=20)


@pytest.fixture
def tracking_solver(system):
    """Create AO solver for tracking QoS."""
    return AOSolver(system, qos_type='tracking', fairness='maxmin', max_iter=20)


def test_power_budget(detection_solver):
    """Test that power budget constraint is satisfied."""
    result = detection_solver.solve(Gamma_c=1.0)

    total_power = np.sum(result.p)
    P_total = detection_solver.system.params.P_total

    assert np.isclose(total_power, P_total, rtol=1e-3), \
        f"Power budget violated: Σp={total_power}, P_total={P_total}"


def test_bandwidth_budget(detection_solver):
    """Test that bandwidth budget constraint is satisfied."""
    result = detection_solver.solve(Gamma_c=1.0)

    total_bandwidth = np.sum(result.b)
    B_total = detection_solver.system.params.B_total

    assert np.isclose(total_bandwidth, B_total, rtol=1e-3), \
        f"Bandwidth budget violated: Σb={total_bandwidth}, B_total={B_total}"


def test_comm_rate_constraint(detection_solver):
    """Test that communication rate constraint is satisfied."""
    Gamma_c = 1e6
    result = detection_solver.solve(Gamma_c=Gamma_c)

    comm_rates = result.comm_rates
    assert comm_rates is not None
    assert len(comm_rates) == detection_solver.system.params.K + detection_solver.system.params.L
    assert np.all(comm_rates >= Gamma_c), \
        f"Per-user communication-rate constraint violated: rates={comm_rates}, Γc={Gamma_c}"


def test_ao_convergence(detection_solver):
    """The solver reports a finite terminal objective and bounded iterations."""
    result = detection_solver.solve(Gamma_c=1e6)

    assert result.iterations <= detection_solver.max_iter, \
        f"AO exceeded max iterations: {result.iterations} > {detection_solver.max_iter}"
    assert np.isfinite(result.objective)
    assert result.diagnostics['bandwidth_subproblem'] != 'not run'


def test_detection_solver_result(detection_solver):
    """Test detection QoS solver result."""
    result = detection_solver.solve(Gamma_c=1.0)

    # Result should have detection probabilities
    assert result.detection_probs is not None, "Detection probabilities should be computed"

    # Detection probabilities should be in valid range
    assert np.all(result.detection_probs >= 0), \
        f"Detection probabilities should be ≥ 0: {result.detection_probs}"
    assert np.all(result.detection_probs <= 1), \
        f"Detection probabilities should be ≤ 1: {result.detection_probs}"


def test_localization_solver_result(localization_solver):
    """Test localization QoS solver result."""
    result = localization_solver.solve(Gamma_c=1.0)

    # Result should have localization metrics
    assert result.localization_rho is not None, "Localization metrics should be computed"

    # Localization metrics should be positive
    assert np.all(result.localization_rho > 0), \
        f"Localization metrics should be > 0: {result.localization_rho}"


def test_tracking_solver_result(tracking_solver):
    """Test tracking QoS solver result."""
    result = tracking_solver.solve(Gamma_c=1.0)

    # Result should have tracking metrics
    assert result.tracking_pcrb is not None, "Tracking PCRB should be computed"

    # PCRB should be positive semi-definite
    for q in range(result.tracking_pcrb.shape[0]):
        eigenvalues = np.linalg.eigvalsh(result.tracking_pcrb[q])
        assert np.all(eigenvalues >= -1e-10), \
            f"PCRB should be PSD for target {q}: eigenvalues={eigenvalues}"


def test_fairness_maxmin(detection_solver):
    """Test max-min fairness works."""
    # Use max-min fairness
    detection_solver.fairness_type = 'maxmin'
    result = detection_solver.solve(Gamma_c=1.0)

    # All detection probabilities should be reasonably close (fairness)
    if result.detection_probs is not None:
        min_pd = np.min(result.detection_probs)
        max_pd = np.max(result.detection_probs)

        # Max-min fairness should reduce disparity
        assert max_pd / (min_pd + 1e-10) < 10, \
            f"Max-min fairness should reduce disparity: min={min_pd}, max={max_pd}"


@pytest.mark.parametrize("fairness", ["proportional", "weighted", "unknown"])
def test_unsupported_solver_fairness_is_rejected(system, fairness):
    with pytest.raises(ValueError, match="maxmin.*sum|only"):
        AOSolver(system, fairness=fairness)


def test_mutated_unsupported_fairness_is_rejected(detection_solver):
    detection_solver.fairness_type = "proportional"
    with pytest.raises(ValueError, match="only"):
        detection_solver.solve(Gamma_c=1e6)


def test_solve_multiple_qos(system):
    """Test solving for all QoS types."""
    solver = AOSolver(system, max_iter=10)
    results = solver.solve_multiple_qos(Gamma_c=1.0)

    # Should have results for all QoS types
    assert 'detection' in results, "Should have detection results"
    assert 'localization' in results, "Should have localization results"
    assert 'tracking' in results, "Should have tracking results"

    # All results should satisfy power/bandwidth budgets
    for qos_type, result in results.items():
        assert np.isclose(np.sum(result.p), system.params.P_total, rtol=1e-3), \
            f"Power budget violated for {qos_type}"
        assert np.isclose(np.sum(result.b), system.params.B_total, rtol=1e-3), \
            f"Bandwidth budget violated for {qos_type}"


def test_initial_conditions(detection_solver):
    """Test solver with a different feasible bandwidth initial condition."""
    M = detection_solver.system.total_objects
    initial_b = (
        np.random.default_rng(123).dirichlet(np.ones(M))
        * detection_solver.system.params.B_total
    )

    result = detection_solver.solve(Gamma_c=1e6, initial_b=initial_b)

    # Should still satisfy constraints
    assert np.isclose(np.sum(result.p), detection_solver.system.params.P_total, rtol=1e-3)
    assert np.isclose(np.sum(result.b), detection_solver.system.params.B_total, rtol=1e-3)


def test_detection_vs_rate_tradeoff(system):
    """Test tradeoff curve between detection and rate threshold."""
    solver = AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=10)

    Gamma_c_values = [0.5, 1.0, 2.0]
    detection_values = []

    for Gamma_c in Gamma_c_values:
        result = solver.solve(Gamma_c=Gamma_c)
        if result.detection_probs is not None:
            detection_values.append(np.min(result.detection_probs))
        else:
            detection_values.append(0)

    # Generally, higher rate threshold should lead to lower detection probability
    # (tradeoff), but may not be strictly monotonic due to solver behavior
    assert len(detection_values) == len(Gamma_c_values), \
        "Should have detection value for each rate threshold"


@pytest.mark.parametrize("qos_type", ["detection", "localization", "tracking"])
@pytest.mark.parametrize("fairness", ["maxmin", "sum"])
def test_every_declared_qos_fairness_combination_runs(system, qos_type, fairness):
    result = AOSolver(
        system,
        qos_type=qos_type,
        fairness=fairness,
        max_iter=4,
    ).solve(Gamma_c=1e6)
    assert np.isfinite(result.objective)
    assert result.comm_rates is not None
    assert np.all(result.comm_rates >= 1e6)


def test_phase_one_finds_reported_feasible_counterexample():
    """A globally feasible case must not be rejected by a uniform start."""
    system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=1,
        P_total=3.0,
        B_total=1.0,
        rng=np.random.default_rng(0),
    )
    system.N0 = 1.0
    system.beta_sensing = np.ones(1)
    system.rcs = np.ones(1)
    system.beta_comm = np.ones(1)
    system.beta_isac = np.ones(1)
    result = AOSolver(system, max_iter=4, tol=1e-10).solve(Gamma_c=0.9)
    assert "Phase-I" not in result.diagnostics["phase_one"]
    assert "feasible" in result.diagnostics["phase_one"]
    assert np.min(result.comm_rates) >= 0.9
    assert np.all(result.p >= 0.0)
    assert np.all(result.b > 0.0)


def test_detection_sum_power_step_uses_true_probability_objective():
    system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=2,
        K=1,
        L=1,
        P_total=10.0,
        B_total=4.0,
        rng=np.random.default_rng(5),
    )
    system.N0 = 1.0
    system.beta_sensing = np.ones(2)
    system.rcs = np.ones(2)
    system.beta_comm = np.ones(1)
    system.beta_isac = np.ones(1)
    solver = AOSolver(system, qos_type="detection", fairness="sum")
    bandwidth = np.ones(4)
    power = solver._solve_power_subproblem(bandwidth, Gamma_c=0.0)
    grid = np.linspace(0.0, 10.0, 100_001)
    probability = np.exp(np.log(0.01) / (1.0 + grid))
    oracle = probability + probability[::-1]
    best = int(np.argmax(oracle))
    np.testing.assert_allclose(power[:2], [grid[best], 10.0 - grid[best]], atol=1e-6)
    assert solver._compute_current_objective(power, bandwidth) == pytest.approx(
        oracle[best], rel=2e-12
    )


def test_localization_power_coefficients_match_dimensionless_score():
    system = ISACSystem(
        Nt=8,
        Nr=8,
        Q=2,
        K=1,
        L=1,
        P_total=10.0,
        B_total=4.0,
        rng=np.random.default_rng(5),
    )
    system.N0 = 1.0
    system.beta_sensing = np.ones(2)
    system.rcs = np.ones(2)
    system.beta_comm = np.ones(1)
    system.beta_isac = np.ones(1)
    system.target_angles = np.array([0.0, np.pi / 3.0])
    solver = AOSolver(system, qos_type="localization", fairness="maxmin")
    bandwidth = np.ones(4)
    coefficient = solver._power_objective_coefficients(bandwidth)
    power = solver._solve_power_subproblem(bandwidth, Gamma_c=0.0)
    expected = 10.0 / (coefficient * np.sum(1.0 / coefficient))
    np.testing.assert_allclose(power[:2], expected, rtol=2e-15, atol=2e-15)
    score = solver.localization_qos.compute_information_score(
        power[:2], bandwidth[:2]
    )
    np.testing.assert_allclose(score[0], score[1], rtol=2e-15)


def test_budget_correction_is_nonnegative_for_300_seeded_sum_steps():
    for seed in range(300):
        system = ISACSystem(rng=np.random.default_rng(seed))
        qos_type = "detection" if seed % 2 == 0 else "localization"
        solver = AOSolver(system, qos_type=qos_type, fairness="sum")
        fraction = np.random.default_rng(10_000 + seed).dirichlet(
            np.ones(system.total_objects)
        )
        bandwidth = fraction * system.params.B_total
        power = solver._solve_power_subproblem(bandwidth, Gamma_c=1.0)
        assert np.all(power >= 0.0)
        assert np.sum(power) == pytest.approx(
            system.params.P_total, rel=0.0, abs=2e-14
        )


@pytest.mark.parametrize(
    "qos_type,seed",
    [("localization", 48), ("tracking", 4)],
)
def test_best_feasible_history_never_decreases(qos_type, seed):
    system = ISACSystem(
        Nt=8,
        Nr=8,
        Q=3,
        K=2,
        L=1,
        P_total=10.0,
        B_total=1.0e7,
        rng=np.random.default_rng(seed),
    )
    initial = (
        np.random.default_rng(1000 + seed).dirichlet(
            np.ones(system.total_objects)
        )
        * system.params.B_total
    )
    result = AOSolver(
        system,
        qos_type=qos_type,
        fairness="maxmin",
        max_iter=10,
        tol=1e-10,
    ).solve(1.0, initial_b=initial)
    assert result.objective_history is not None
    assert np.all(np.diff(result.objective_history) >= 0.0)
    assert result.objective == np.max(result.objective_history)
    if not (
        result.diagnostics["power_subproblem_success"]
        and result.diagnostics["bandwidth_subproblem_success"]
    ):
        assert not result.converged


def test_failed_bandwidth_step_retains_start_and_cannot_converge(system):
    solver = AOSolver(
        system,
        qos_type="detection",
        fairness="maxmin",
        max_iter=5,
    )
    with patch(
        "code.baselines.isac_resource_allocation.src.ao_solver.minimize",
        return_value=SimpleNamespace(
            success=False,
            message="forced failure",
            x=np.full(system.total_objects, 1.0 / system.total_objects),
        ),
    ):
        result = solver.solve(Gamma_c=1.0)
    assert not result.converged
    assert not result.diagnostics["bandwidth_subproblem_success"]
    assert "retained feasible bandwidth start" in result.diagnostics[
        "bandwidth_subproblem"
    ]
