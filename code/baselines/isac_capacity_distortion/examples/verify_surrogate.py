#!/usr/bin/env python3
"""Emit a machine-readable certificate for the local numerical surrogate."""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np

from ..src.optimization import (
    covariance_shaping_surrogate,
    isotropic_covariance,
    water_filling_covariance,
)
from ..src.system_model import (
    GaussianISACChannel,
    angle_to_channel,
    compute_bfim,
    compute_crb,
    compute_phi_angle,
    compute_rate,
    compute_rate_per_symbol,
)


SEED = 20260717
GRID_POINTS = 500_001


def _diagonal_power(covariance: np.ndarray) -> np.ndarray:
    off_diagonal = covariance - np.diag(np.diag(covariance))
    if np.linalg.norm(off_diagonal) > 2e-12:
        raise AssertionError("diagonal-channel oracle received nondiagonal covariance")
    return np.real(np.diag(covariance))


def build_certificate() -> dict[str, object]:
    """Run independent analytic/numeric checks and return their evidence."""

    channel = np.diag([2.0, 0.75]).astype(np.complex128)
    sigma_c2 = 0.8
    power_per_tx = 1.0
    budget = power_per_tx * channel.shape[1]
    grid = np.linspace(0.0, budget, GRID_POINTS)
    grid_step = float(grid[1] - grid[0])
    gains = np.array([4.0, 0.75**2]) / sigma_c2
    grid_rates = np.log1p(gains[0] * grid) + np.log1p(
        gains[1] * (budget - grid)
    )
    grid_rate_index = int(np.argmax(grid_rates))
    grid_rate_power = float(grid[grid_rate_index])
    grid_rate_value = float(grid_rates[grid_rate_index])

    water_filled = water_filling_covariance(
        power_per_tx,
        channel,
        sigma_c2,
    )
    water_power = _diagonal_power(water_filled)
    water_rate = compute_rate(water_filled, channel, sigma_c2)
    water_power_error = abs(float(water_power[0]) - grid_rate_power)
    water_rate_gap = grid_rate_value - water_rate

    alpha = 0.37
    surrogate = covariance_shaping_surrogate(
        alpha,
        power_per_tx,
        channel,
        sigma_c2,
    )
    surrogate_power = _diagonal_power(surrogate)
    interior_step = budget / (GRID_POINTS + 1)
    interior = np.linspace(
        interior_step,
        budget - interior_step,
        GRID_POINTS,
    )
    surrogate_grid_objective = (
        -(1 - alpha) * (np.log(interior) + np.log(budget - interior))
        - alpha
        * (
            np.log1p(gains[0] * interior)
            + np.log1p(gains[1] * (budget - interior))
        )
    )
    surrogate_index = int(np.argmin(surrogate_grid_objective))
    surrogate_grid_power = float(interior[surrogate_index])
    surrogate_objective = float(
        -(1 - alpha) * np.sum(np.log(surrogate_power))
        - alpha * compute_rate(surrogate, channel, sigma_c2)
    )
    surrogate_grid_value = float(surrogate_grid_objective[surrogate_index])
    surrogate_power_error = abs(
        float(surrogate_power[0]) - surrogate_grid_power
    )
    surrogate_objective_gap = surrogate_objective - surrogate_grid_value
    kkt_values = (
        (1 - alpha) / surrogate_power
        + alpha * gains / (1 + gains * surrogate_power)
    )
    kkt_spread = float(np.max(kkt_values) - np.min(kkt_values))

    covariance = np.diag([0.25, 2.0]).astype(np.complex128)
    prior = np.diag([1.0, 3.0]).astype(np.complex128)
    bfim = compute_bfim(covariance, T=7, sigma_s2=0.5, Jp=prior)
    bfim_expected = np.diag([4.5, 31.0]).astype(np.complex128)
    crb = compute_crb(bfim=bfim)
    crb_expected = 1 / 4.5 + 1 / 31.0
    bfim_error = float(np.max(np.abs(bfim - bfim_expected)))
    crb_error = abs(crb - crb_expected)
    weak_crb = compute_crb(bfim=np.array([[1e-13]], dtype=np.complex128))

    large_bfim = float(compute_bfim(np.array([[1e308]]), 1, 1)[0, 0].real)
    large_bfim_relative_error = abs(large_bfim - 1e308) / 1e308
    weak_scale_noise = np.nextafter(0.0, 1.0)
    weak_scale_bfim = float(
        compute_bfim(np.array([[1e-308]]), 2, weak_scale_noise)[0, 0].real
    )
    weak_scale_bfim_expected = float(2e-308 / weak_scale_noise)
    weak_scale_bfim_relative_error = float(
        abs(weak_scale_bfim - weak_scale_bfim_expected)
        / weak_scale_bfim_expected
    )
    rejected_oversized_bfim = False
    try:
        compute_bfim(np.array([[1e308]]), 2, weak_scale_noise)
    except ValueError:
        rejected_oversized_bfim = True

    siso_channel = np.array([[1.2 - 0.4j]], dtype=np.complex128)
    siso_covariance = np.array([[0.7]], dtype=np.complex128)
    siso_rate = compute_rate(siso_covariance, siso_channel, 0.3)
    siso_expected = float(np.log1p((1.2**2 + 0.4**2) * 0.7 / 0.3))
    siso_error = abs(siso_rate - siso_expected)
    received_waveform_rate = compute_rate_per_symbol(
        np.array([[1e200 + 0j]]),
        np.array([[1e-200 + 0j]]),
        1.0,
    )
    received_waveform_rate_error = float(
        abs(received_waveform_rate - np.log(2.0))
    )
    cancellation_cases = (
        ([1e280, 1.0, 1e280], [1.0, 1e-60, -1.0]),
        ([1.0, 1e280, 1e280], [1e-60, 1.0, -1.0]),
        ([1e280, 1e280, 1.0], [1.0, -1.0, 1e-60]),
    )
    cancellation_rates = [
        compute_rate_per_symbol(
            np.asarray(waveform_values, dtype=np.complex128).reshape(3, 1),
            np.asarray(channel_values, dtype=np.complex128).reshape(1, 3),
            1.0,
        )
        for channel_values, waveform_values in cancellation_cases
    ]
    cancellation_rate_expected = float(np.log1p(1e-120))
    cancellation_rate_relative_error = max(
        abs(value - cancellation_rate_expected) / cancellation_rate_expected
        for value in cancellation_rates
    )

    dynamic_channel = np.diag([2e-200, 1e-200]).astype(np.complex128)
    dynamic_covariance = water_filling_covariance(
        1.0,
        dynamic_channel,
        1e-300,
    )
    dynamic_covariance_expected = np.diag([2.0, 0.0]).astype(np.complex128)
    dynamic_covariance_error = float(
        np.max(np.abs(dynamic_covariance - dynamic_covariance_expected))
    )
    dynamic_rate = compute_rate(dynamic_covariance, dynamic_channel, 1e-300)
    dynamic_rate_expected = float(np.log1p(8e-100))
    dynamic_rate_relative_error = abs(
        dynamic_rate - dynamic_rate_expected
    ) / dynamic_rate_expected

    subnormal_channel = np.diag([2e-155, 1e-155]).astype(np.complex128)
    subnormal_covariance = water_filling_covariance(1.0, subnormal_channel, 1.0)
    subnormal_covariance_error = float(
        np.max(np.abs(subnormal_covariance - dynamic_covariance_expected))
    )
    subnormal_rate = compute_rate(subnormal_covariance, subnormal_channel, 1.0)
    subnormal_rate_expected = float(np.log1p(8e-310))
    subnormal_rate_relative_error = abs(
        subnormal_rate - subnormal_rate_expected
    ) / subnormal_rate_expected

    extreme_surrogate = covariance_shaping_surrogate(
        0.5,
        1.0,
        np.array([[1.0e154]], dtype=np.complex128),
        1.0,
    )
    extreme_surrogate_error = float(abs(extreme_surrogate[0, 0] - 1.0))
    mixed_extreme_surrogate = covariance_shaping_surrogate(
        0.5,
        1.0,
        np.diag([1.0e154, 0.0]).astype(np.complex128),
        1.0,
    )
    mixed_extreme_expected = np.diag([4 / 3, 2 / 3])
    mixed_extreme_surrogate_error = float(
        np.max(np.abs(mixed_extreme_surrogate - mixed_extreme_expected))
    )

    overflow_crb = compute_crb(bfim=np.array([[1e-320]], dtype=np.complex128))
    rejected_oversized_water_budget = False
    rejected_oversized_surrogate_budget = False
    try:
        water_filling_covariance(1e308, np.eye(2), 1.0)
    except ValueError:
        rejected_oversized_water_budget = True
    try:
        covariance_shaping_surrogate(0.5, 1e308, np.eye(2), 1.0)
    except ValueError:
        rejected_oversized_surrogate_budget = True

    theta = np.deg2rad(27.0)
    angle_covariance = np.diag([0.2, 0.5, 0.8, 1.1]).astype(np.complex128)
    epsilon = 1e-6
    derivative = (
        angle_to_channel(theta + epsilon, 4, 3)
        - angle_to_channel(theta - epsilon, 4, 3)
    ) / (2 * epsilon)
    numeric_information = 2 * float(
        np.real(
            np.trace(
                derivative.conj().T @ derivative @ angle_covariance
            )
        )
    )
    analytic_information = float(
        compute_phi_angle(angle_covariance, 9, theta, 4, 3)[0, 0].real
    )
    derivative_relative_error = abs(
        analytic_information - numeric_information
    ) / numeric_information
    rejected_unrepresentable_geometry = False
    try:
        compute_phi_angle(
            np.eye(2),
            1,
            0.0,
            2,
            2,
            d_tx=1e308,
            d_rx=1e308,
        )
    except ValueError:
        rejected_unrepresentable_geometry = True

    def sensing(parameter: np.ndarray) -> np.ndarray:
        return np.eye(2, dtype=np.complex128) * float(parameter[0])

    waveform = np.zeros((2, 6), dtype=np.complex128)
    first_channel = GaussianISACChannel(
        np.eye(2),
        sensing,
        0.4,
        0.6,
        2,
        2,
        2,
        6,
        rng=np.random.default_rng(SEED),
    )
    second_channel = GaussianISACChannel(
        np.eye(2),
        sensing,
        0.4,
        0.6,
        2,
        2,
        2,
        6,
        rng=np.random.default_rng(SEED),
    )
    repeatability_error = float(
        np.max(
            np.abs(
                first_channel.comm_receive(waveform)
                - second_channel.comm_receive(waveform)
            )
        )
    )

    isotropic = isotropic_covariance(power_per_tx, 2)
    constraints = {
        "water_trace_error": abs(float(np.trace(water_filled).real) - budget),
        "water_min_eigenvalue": float(np.min(np.linalg.eigvalsh(water_filled))),
        "surrogate_trace_error": abs(float(np.trace(surrogate).real) - budget),
        "surrogate_min_eigenvalue": float(np.min(np.linalg.eigvalsh(surrogate))),
        "water_rate_minus_isotropic": water_rate
        - compute_rate(isotropic, channel, sigma_c2),
    }

    tolerances = {
        "grid_steps": 1.1,
        "objective_absolute": 1e-10,
        "kkt_absolute": 2e-12,
        "analytic_absolute": 2e-14,
        "bfim_scale_relative": 5e-15,
        "cancellation_rate_relative": 2e-15,
        "steering_derivative_relative": 1e-8,
        "dynamic_range_relative": 2e-15,
        "subnormal_gain_relative": 1e-12,
        "constraint_absolute": 2e-12,
    }
    checks = {
        "water_grid_power": water_power_error
        <= tolerances["grid_steps"] * grid_step,
        "water_grid_objective": abs(water_rate_gap)
        <= tolerances["objective_absolute"],
        "surrogate_grid_power": surrogate_power_error
        <= tolerances["grid_steps"] * interior_step,
        "surrogate_grid_objective": abs(surrogate_objective_gap)
        <= tolerances["objective_absolute"],
        "surrogate_kkt": kkt_spread <= tolerances["kkt_absolute"],
        "bfim_analytic": bfim_error <= tolerances["analytic_absolute"],
        "bfim_scale_safe": (
            large_bfim_relative_error <= tolerances["bfim_scale_relative"]
            and weak_scale_bfim_relative_error
            <= tolerances["bfim_scale_relative"]
        ),
        "crb_analytic": crb_error <= tolerances["analytic_absolute"],
        "weak_information_is_finite": np.isfinite(weak_crb)
        and abs(weak_crb - 1e13) <= 1.0,
        "siso_rate_analytic": siso_error <= tolerances["analytic_absolute"],
        "received_waveform_rate_scale_safe": received_waveform_rate_error
        <= tolerances["analytic_absolute"],
        "received_waveform_cancellation_permutation_invariant": (
            min(cancellation_rates) > 0.0
            and cancellation_rate_relative_error
            <= tolerances["cancellation_rate_relative"]
        ),
        "dynamic_range_scaling": (
            dynamic_covariance_error <= tolerances["analytic_absolute"]
            and dynamic_rate > 0
            and dynamic_rate_relative_error
            <= tolerances["dynamic_range_relative"]
        ),
        "subnormal_gain_scaling": (
            subnormal_covariance_error <= tolerances["analytic_absolute"]
            and subnormal_rate > 0
            and subnormal_rate_relative_error
            <= tolerances["subnormal_gain_relative"]
        ),
        "surrogate_discriminant_scale_safe": (
            extreme_surrogate_error == 0.0
        ),
        "surrogate_power_gain_product_scale_safe": (
            mixed_extreme_surrogate_error
            <= tolerances["analytic_absolute"]
        ),
        "unrepresentable_outputs_are_explicit": (
            np.isinf(overflow_crb)
            and rejected_oversized_bfim
            and rejected_oversized_water_budget
            and rejected_oversized_surrogate_budget
            and rejected_unrepresentable_geometry
        ),
        "steering_derivative": derivative_relative_error
        <= tolerances["steering_derivative_relative"],
        "seed_repeatability": repeatability_error == 0,
        "physical_postconditions": (
            constraints["water_trace_error"] <= tolerances["constraint_absolute"]
            and constraints["surrogate_trace_error"]
            <= tolerances["constraint_absolute"]
            and constraints["water_min_eigenvalue"]
            >= -tolerances["constraint_absolute"]
            and constraints["surrogate_min_eigenvalue"]
            >= -tolerances["constraint_absolute"]
            and constraints["water_rate_minus_isotropic"] >= -1e-13
        ),
    }

    return {
        "schema_version": 1,
        "baseline": "isac_capacity_distortion",
        "evidence_level": "educational-surrogate",
        "paper_figure_parity": False,
        "status": "pass" if all(checks.values()) else "fail",
        "claim": {
            "level": "educational-surrogate",
            "paper_figure_parity": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seed": SEED,
        },
        "oracles": {
            "water_filling_dense_grid": {
                "grid_points": GRID_POINTS,
                "grid_step": grid_step,
                "solver_first_mode_power": float(water_power[0]),
                "grid_first_mode_power": grid_rate_power,
                "power_error": water_power_error,
                "rate_gap": water_rate_gap,
            },
            "surrogate_dense_grid": {
                "alpha": alpha,
                "grid_points": GRID_POINTS,
                "grid_step": interior_step,
                "solver_first_mode_power": float(surrogate_power[0]),
                "grid_first_mode_power": surrogate_grid_power,
                "power_error": surrogate_power_error,
                "objective_gap": surrogate_objective_gap,
                "kkt_spread": kkt_spread,
            },
            "analytic_formulas": {
                "bfim_max_error": bfim_error,
                "crb_error": crb_error,
                "weak_information_crb": weak_crb,
                "siso_rate_error": siso_error,
            },
            "bfim_scaling": {
                "large_information_value": large_bfim,
                "large_information_relative_error": large_bfim_relative_error,
                "weak_information_value": weak_scale_bfim,
                "weak_information_expected": weak_scale_bfim_expected,
                "weak_information_relative_error": weak_scale_bfim_relative_error,
            },
            "received_waveform": {
                "waveform_scale": 1e200,
                "channel_scale": 1e-200,
                "computed_rate": received_waveform_rate,
                "expected_rate": float(np.log(2.0)),
                "absolute_error": received_waveform_rate_error,
                "cross_scale_cancellation_rates": cancellation_rates,
                "cross_scale_cancellation_expected": (
                    cancellation_rate_expected
                ),
                "cross_scale_cancellation_max_relative_error": (
                    cancellation_rate_relative_error
                ),
            },
            "dynamic_range": {
                "channel_scale": 1e-200,
                "noise_variance": 1e-300,
                "covariance_max_error": dynamic_covariance_error,
                "computed_rate": dynamic_rate,
                "expected_rate": dynamic_rate_expected,
                "rate_relative_error": dynamic_rate_relative_error,
            },
            "subnormal_gain": {
                "channel_scale": 1e-155,
                "covariance_max_error": subnormal_covariance_error,
                "computed_rate": subnormal_rate,
                "expected_rate": subnormal_rate_expected,
                "rate_relative_error": subnormal_rate_relative_error,
            },
            "surrogate_discriminant_scaling": {
                "channel_amplitude": 1.0e154,
                "alpha": 0.5,
                "computed_covariance": float(extreme_surrogate[0, 0].real),
                "unit_trace_oracle_absolute_error": extreme_surrogate_error,
            },
            "surrogate_power_gain_product_scaling": {
                "channel_amplitudes": [1.0e154, 0.0],
                "power_budget": 2.0,
                "computed_diagonal": [
                    float(mixed_extreme_surrogate[0, 0].real),
                    float(mixed_extreme_surrogate[1, 1].real),
                ],
                "closed_form_diagonal": [4 / 3, 2 / 3],
                "maximum_absolute_error": mixed_extreme_surrogate_error,
            },
            "explicit_range_limits": {
                "overflow_crb_is_infinite": bool(np.isinf(overflow_crb)),
                "oversized_bfim_rejected": rejected_oversized_bfim,
                "oversized_water_budget_rejected": (
                    rejected_oversized_water_budget
                ),
                "oversized_surrogate_budget_rejected": (
                    rejected_oversized_surrogate_budget
                ),
                "unrepresentable_angle_geometry_rejected": (
                    rejected_unrepresentable_geometry
                ),
            },
            "finite_difference": {
                "angle_information_relative_error": derivative_relative_error,
            },
            "repeatability": {
                "maximum_sample_error": repeatability_error,
            },
        },
        "constraints": constraints,
        "tolerances": tolerances,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit compact JSON instead of indented JSON",
    )
    arguments = parser.parse_args()
    certificate = build_certificate()
    print(
        json.dumps(
            certificate,
            allow_nan=False,
            indent=None if arguments.json else 2,
            sort_keys=True,
        )
    )
    if certificate["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
