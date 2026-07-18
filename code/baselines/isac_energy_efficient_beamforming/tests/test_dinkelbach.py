"""Strict tests for the exact fixed-direction Dinkelbach slice."""

import numpy as np
import pytest

from ..src.dinkelbach_solver import (
    InfeasibleReferenceProblem,
    SingleUserPowerDinkelbach,
)
from ..src.system_model import ISACSystemModel


def _solver() -> SingleUserPowerDinkelbach:
    model = ISACSystemModel(
        M=4,
        K=1,
        N=5,
        P_max_dbm=30.0,
        P0_dbm=30.0,
        sigma_c_dbm=-20.0,
        sigma_s_dbm=-10.0,
        seed=91,
    )
    return SingleUserPowerDinkelbach(model, tolerance=1e-12)


def test_dinkelbach_matches_independent_dense_grid_oracle() -> None:
    solver = _solver()
    result = solver.solve()
    oracle_power, oracle_ee, spacing = solver.dense_grid_oracle()
    assert result.converged
    assert abs(result.power_watt - oracle_power) <= 1.1 * spacing
    assert result.ee_c >= oracle_ee * (1.0 - 2e-10)
    assert abs(result.residual) <= 1e-10


def test_history_is_a_real_dinkelbach_certificate() -> None:
    result = _solver().solve()
    lambdas = np.array([item.lambda_before for item in result.history])
    residuals = np.array([item.subtractive_residual for item in result.history])
    assert len(result.history) >= 2
    assert residuals[0] > 0.0
    assert np.all(np.diff(lambdas) >= -1e-13)
    assert abs(residuals[-1]) < abs(residuals[0]) * 1e-8


def test_sinr_constraint_is_exactly_converted_to_minimum_power() -> None:
    solver = _solver()
    gain = abs(solver.model.get_channel(0).conj() @ solver.direction) ** 2
    required_power = 0.8 * solver.model.P_max
    gamma = required_power * gain / solver.model.sigma_c2
    p_min, _ = solver.feasible_power_interval(gamma_min=gamma)
    assert p_min == pytest.approx(required_power, rel=1e-13)
    result = solver.solve(gamma_min=gamma)
    assert result.power_watt >= required_power * (1.0 - 1e-12)
    assert result.sinr >= gamma * (1.0 - 1e-12)


def test_crb_constraint_is_postvalidated() -> None:
    solver = _solver()
    unconstrained = solver.solve()
    target = unconstrained.crb / 2.0
    p_min, _ = solver.feasible_power_interval(crb_max=target)
    result = solver.solve(crb_max=target)
    assert result.power_watt >= p_min * (1.0 - 1e-12)
    assert result.crb <= target * (1.0 + 1e-10)


def test_infeasible_constraint_raises_instead_of_returning_old_iterate() -> None:
    solver = _solver()
    gain = abs(solver.model.get_channel(0).conj() @ solver.direction) ** 2
    impossible_gamma = 2.0 * solver.model.P_max * gain / solver.model.sigma_c2
    with pytest.raises(InfeasibleReferenceProblem, match="exceeds"):
        solver.solve(gamma_min=impossible_gamma)


def test_same_seed_yields_bitwise_equal_result() -> None:
    first = _solver().solve()
    second = _solver().solve()
    assert first.power_watt == second.power_watt
    assert first.ee_c == second.ee_c
    np.testing.assert_array_equal(first.W, second.W)


def test_multuser_problem_is_rejected_as_out_of_scope() -> None:
    model = ISACSystemModel(M=4, K=2, N=5)
    with pytest.raises(ValueError, match="K=1"):
        SingleUserPowerDinkelbach(model)
