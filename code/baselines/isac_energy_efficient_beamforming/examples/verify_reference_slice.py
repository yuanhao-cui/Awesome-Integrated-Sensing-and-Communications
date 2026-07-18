"""Emit a machine-checkable certificate for the validated model slice."""

from __future__ import annotations

import argparse
from itertools import permutations
import json

import numpy as np

from ..src.dinkelbach_solver import SingleUserPowerDinkelbach
from ..src.ee_metrics import (
    compute_crb,
    compute_sinr,
    compute_sum_rate,
    compute_total_power,
)
from ..src.system_model import ISACSystemModel
from ..src.quadratic_transform import compute_sum_rate_quadratic


def build_certificate() -> dict[str, object]:
    """Run independent Dinkelbach and FIM comparisons."""

    model = ISACSystemModel(
        M=4,
        K=1,
        N=5,
        L=30,
        P_max_dbm=30.0,
        P0_dbm=30.0,
        epsilon=0.35,
        sigma_c_dbm=-20.0,
        sigma_s_dbm=-10.0,
        seed=20260717,
    )
    solver = SingleUserPowerDinkelbach(model, tolerance=1e-12)
    result = solver.solve(target_angle_deg=90.0)
    oracle_power, oracle_ee, spacing = solver.dense_grid_oracle(
        target_angle_deg=90.0,
        n_points=500_001,
    )

    theta = np.pi / 2.0
    a_t = model.steering_vector_tx(theta)
    a_r = model.steering_vector_rx(theta)
    da_t = model.steering_derivative_tx(theta)
    da_r = model.steering_derivative_rx(theta)
    covariance_crb = compute_crb(
        result.W,
        a_t,
        a_r,
        da_t,
        da_r,
        model.sigma_s2,
        model.L,
    )

    # With K=1, the all-one symbol row satisfies S S^H = L exactly.
    symbols = np.ones((1, model.L), dtype=complex)
    X = result.W @ symbols
    response = np.outer(a_r, a_t.conj())
    response_derivative = (
        np.outer(da_r, a_t.conj()) + np.outer(a_r, da_t.conj())
    )
    g = (response @ X).reshape(-1, order="F")
    g_dot = (response_derivative @ X).reshape(-1, order="F")
    effective_information = (
        np.vdot(g_dot, g_dot).real
        - abs(np.vdot(g, g_dot)) ** 2 / np.vdot(g, g).real
    )
    explicit_crb = model.sigma_s2 / (2.0 * effective_information)
    crb_relative_error = abs(covariance_crb - explicit_crb) / explicit_crb

    low_scale = 1.0e-8
    low_scale_crb = compute_crb(
        low_scale * result.W,
        a_t,
        a_r,
        da_t,
        da_r,
        model.sigma_s2,
        model.L,
    )
    low_scale_oracle = explicit_crb / low_scale**2
    low_scale_crb_relative_error = (
        abs(low_scale_crb - low_scale_oracle) / low_scale_oracle
    )

    low_snr_rate_oracle = 1.4426950408889633e-20
    low_snr_rate = compute_sum_rate(
        np.array([[1.0e-10 + 0.0j]]),
        np.array([[1.0 + 0.0j]]),
        sigma_c2=1.0,
    )
    low_snr_rate_relative_error = abs(
        low_snr_rate - low_snr_rate_oracle
    ) / low_snr_rate_oracle

    extreme_quadratic_rate_oracle = float(
        np.log1p(1.0e308) / np.log(2.0)
    )
    extreme_quadratic_rate = compute_sum_rate_quadratic(
        np.array([[1.0e154 + 0.0j]]),
        np.array([[1.0 + 0.0j]]),
        1.0,
    )
    extreme_quadratic_rate_relative_error = abs(
        extreme_quadratic_rate - extreme_quadratic_rate_oracle
    ) / extreme_quadratic_rate_oracle

    cancellation_channel = np.array([1.0, 0.0])
    cancellation_beams = np.array(
        [[1.0, 1.0e-10], [0.0, 0.0]]
    )
    cancellation_noise = 1.0e-30
    weak_interference_sinr_oracle = 9.999999999000001e19
    weak_interference_sinr = compute_sinr(
        0,
        cancellation_channel,
        cancellation_beams,
        cancellation_noise,
    )
    weak_interference_sinr_relative_error = abs(
        weak_interference_sinr - weak_interference_sinr_oracle
    ) / weak_interference_sinr_oracle
    weak_interference_sum_rate_oracle = 66.43856189760298
    weak_interference_sum_rate = compute_sum_rate(
        np.eye(2), cancellation_beams, cancellation_noise
    )
    weak_interference_rate_relative_error = abs(
        weak_interference_sum_rate - weak_interference_sum_rate_oracle
    ) / weak_interference_sum_rate_oracle

    power_fixture = np.array([[0.6 + 0.8j]])
    power_scale = 1.0e150
    base_power = compute_total_power(power_fixture)
    scaled_power = compute_total_power(power_scale * power_fixture)
    power_scale_oracle = 1.0e300
    power_scale_relative_error = abs(
        scaled_power - power_scale_oracle
    ) / power_scale_oracle

    cancellation_tail_channel = np.array(
        [1.0e280, 1.0e280, 1.0e-60]
    )
    cancellation_tail_beam = np.array([[1.0], [-1.0], [1.0]])
    cancellation_tail_sinr_oracle = 1.0e-120
    cancellation_tail_rate_oracle = 1.4426950408889635e-120
    cancellation_tail_sinr_values: list[float] = []
    cancellation_tail_rate_values: list[float] = []
    for permutation in permutations(range(3)):
        indices = np.asarray(permutation)
        cancellation_tail_sinr_values.append(
            compute_sinr(
                0,
                cancellation_tail_channel[indices],
                cancellation_tail_beam[indices],
                1.0,
            )
        )
        cancellation_tail_rate_values.append(
            compute_sum_rate(
                cancellation_tail_channel[indices][None, :],
                cancellation_tail_beam[indices],
                1.0,
            )
        )
    cancellation_tail_sinr_error = max(
        abs(value - cancellation_tail_sinr_oracle)
        / cancellation_tail_sinr_oracle
        for value in cancellation_tail_sinr_values
    )
    cancellation_tail_rate_error = max(
        abs(value - cancellation_tail_rate_oracle)
        / cancellation_tail_rate_oracle
        for value in cancellation_tail_rate_values
    )

    derivative_step = 1e-6
    finite_difference = (
        model.steering_vector_tx(theta + derivative_step)
        - model.steering_vector_tx(theta - derivative_step)
    ) / (2.0 * derivative_step)
    derivative_relative_error = float(
        np.linalg.norm(da_t - finite_difference) / np.linalg.norm(da_t)
    )

    allowed_power_error = 1.1 * spacing
    crb_tolerance = 5e-12
    derivative_tolerance = 1e-8
    checks = {
        "dinkelbach_converged": bool(result.converged),
        "dinkelbach_matches_grid": bool(
            abs(result.power_watt - oracle_power) <= allowed_power_error
        ),
        "crb_matches_explicit_fim": bool(
            crb_relative_error <= crb_tolerance
        ),
        "crb_scale_invariance_at_1e_minus_8": bool(
            np.isfinite(low_scale_crb)
            and low_scale_crb_relative_error <= crb_tolerance
        ),
        "sub_epsilon_snr_rate_preserved": bool(
            low_snr_rate_relative_error <= 5e-16
        ),
        "quadratic_optimum_reduction_is_scale_safe": bool(
            extreme_quadratic_rate_relative_error <= 2e-15
        ),
        "weak_interference_not_cancelled": bool(
            weak_interference_sinr_relative_error <= 3e-15
        ),
        "weak_interference_sum_rate": bool(
            weak_interference_rate_relative_error <= 3e-15
        ),
        "total_power_scale_oracle": bool(
            abs(base_power - 1.0) <= 3e-15
            and power_scale_relative_error <= 2e-15
        ),
        "cancellation_tail_across_340_decades": bool(
            cancellation_tail_sinr_error <= 3e-15
            and cancellation_tail_rate_error <= 3e-15
        ),
        "cancellation_tail_permutation_invariant": bool(
            len(set(cancellation_tail_sinr_values)) == 1
            and len(set(cancellation_tail_rate_values)) == 1
        ),
        "steering_derivative_matches_finite_difference": (
            bool(derivative_relative_error <= derivative_tolerance)
        ),
        "power_constraint": bool(
            result.power_watt <= model.P_max * (1.0 + 1e-12)
        ),
    }
    return {
        "schema_version": 1,
        "baseline": "isac_energy_efficient_beamforming",
        "evidence_level": "equation-level",
        "paper_figure_parity": False,
        "status": "pass" if all(checks.values()) else "fail",
        "claim": "equation-level-single-user-fixed-direction",
        "reference": {
            "doi": "10.1109/TCOMM.2024.3369696",
            "equations": [2, 4, 6, 7, 9, 17],
            "paper_figure_parity_claimed": False,
        },
        "parameters": {
            "paper_reported": {
                "N": 20,
                "L": 30,
                "P_max_dbm": 30.0,
                "epsilon": 0.35,
                "theta_deg": 90.0,
            },
            "local_declared": {
                "M": model.M,
                "K": model.K,
                "N": model.N,
                "P0_dbm": 30.0,
                "sigma_c_dbm": -20.0,
                "sigma_s_dbm": -10.0,
                "channel": "seeded CN(0,1)",
                "seed": model.seed,
            },
        },
        "dinkelbach": {
            "power_watt": result.power_watt,
            "grid_oracle_power_watt": oracle_power,
            "power_error_watt": abs(result.power_watt - oracle_power),
            "allowed_power_error_watt": allowed_power_error,
            "energy_efficiency": result.ee_c,
            "grid_oracle_energy_efficiency": oracle_ee,
            "subtractive_residual": result.residual,
            "iterations": result.n_iterations,
        },
        "crb": {
            "covariance_form": covariance_crb,
            "explicit_snapshot_fim": explicit_crb,
            "relative_error": crb_relative_error,
            "relative_tolerance": crb_tolerance,
            "low_beam_scale": low_scale,
            "low_scale_value": low_scale_crb,
            "low_scale_inverse_square_oracle": low_scale_oracle,
            "low_scale_relative_error": low_scale_crb_relative_error,
        },
        "low_snr_rate": {
            "sinr": 1.0e-20,
            "value": low_snr_rate,
            "log1p_oracle": low_snr_rate_oracle,
            "relative_error": low_snr_rate_relative_error,
            "relative_tolerance": 5e-16,
        },
        "quadratic_transform_extreme_projection": {
            "channel_amplitude": 1.0e154,
            "sinr": 1.0e308,
            "value": extreme_quadratic_rate,
            "log1p_oracle": extreme_quadratic_rate_oracle,
            "relative_error": extreme_quadratic_rate_relative_error,
            "relative_tolerance": 2.0e-15,
        },
        "weak_interference": {
            "desired_projection_magnitude": 1.0,
            "interference_projection_magnitude": 1.0e-10,
            "noise_power": cancellation_noise,
            "sinr": weak_interference_sinr,
            "sinr_oracle": weak_interference_sinr_oracle,
            "sinr_relative_error": weak_interference_sinr_relative_error,
            "sum_rate": weak_interference_sum_rate,
            "sum_rate_oracle": weak_interference_sum_rate_oracle,
            "sum_rate_relative_error": weak_interference_rate_relative_error,
            "relative_tolerance": 3e-15,
        },
        "total_power_scale": {
            "base_power": base_power,
            "common_amplitude_scale": power_scale,
            "scaled_power": scaled_power,
            "square_scale_oracle": power_scale_oracle,
            "relative_error": power_scale_relative_error,
            "relative_tolerance": 2e-15,
        },
        "exact_cancellation_tail": {
            "large_projection_terms": [1.0e280, -1.0e280],
            "residual_projection": 1.0e-60,
            "antenna_permutations_checked": len(
                cancellation_tail_sinr_values
            ),
            "sinr_values": cancellation_tail_sinr_values,
            "sinr_oracle": cancellation_tail_sinr_oracle,
            "sinr_max_relative_error": cancellation_tail_sinr_error,
            "rate_values": cancellation_tail_rate_values,
            "rate_oracle": cancellation_tail_rate_oracle,
            "rate_max_relative_error": cancellation_tail_rate_error,
            "relative_tolerance": 3e-15,
        },
        "steering_derivative": {
            "relative_error": derivative_relative_error,
            "relative_tolerance": derivative_tolerance,
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit compact JSON for the automated gate",
    )
    args = parser.parse_args()
    certificate = build_certificate()
    if args.json:
        print(json.dumps(certificate, sort_keys=True, allow_nan=False))
    else:
        print(
            json.dumps(
                certificate,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
