"""Emit a machine-checkable certificate for the declared RIS surrogate."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..src.ao_solver import AlternatingOptimizationSolver
from ..src.numerics import stable_triangle_sensing_snr
from ..src.ris_phase import RISPhaseOptimizer
from ..src.system_model import RIS_ISAC_System


def build_certificate() -> dict[str, object]:
    """Check stream statistics, monotonic phases, and physical constraints."""

    seed = 20260717
    oracle_system = RIS_ISAC_System(
        M=4, K=2, L=12, noise_power=1.0, seed=seed
    )
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(oracle_system.M, oracle_system.K)) + 1j * rng.normal(
        size=(oracle_system.M, oracle_system.K)
    )
    W *= np.sqrt(oracle_system.P_max / np.vdot(W, W).real)
    before_snr = oracle_system.compute_snr_sensing(W)
    theta, achieved_snr = RISPhaseOptimizer(oracle_system).optimize_for_snr(W)

    low_scale = 1.0e-20
    low_scale_system = RIS_ISAC_System(
        M=4, K=2, L=12, noise_power=1.0, seed=seed
    )
    low_scale_system.channels["a_bs"] *= low_scale
    low_scale_system.channels["H_BR"] *= low_scale
    low_before_snr = low_scale_system.compute_snr_sensing(W)
    low_theta, low_achieved_snr = RISPhaseOptimizer(
        low_scale_system
    ).optimize_for_snr(W)
    low_scale_oracle = achieved_snr * low_scale**2
    low_scale_relative_error = abs(
        low_achieved_snr - low_scale_oracle
    ) / low_scale_oracle
    phase_scale_error = float(np.max(np.abs(theta - low_theta)))

    stream_system = RIS_ISAC_System(
        M=1, K=2, L=1, noise_power=1.0, seed=seed
    )
    stream_system.channels["a_bs"] = np.ones(1, dtype=complex)
    stream_system.channels["a_ris"] = np.zeros(1, dtype=complex)
    stream_system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
    opposite_stream_snr = stream_system.compute_snr_sensing(
        np.array([[1.0, -1.0]], dtype=complex)
    )
    equal_stream_snr = stream_system.compute_snr_sensing(
        np.array([[1.0, 1.0]], dtype=complex)
    )

    linearization_reference = np.array([1.0, -1.0], dtype=complex)
    linearization_trial = np.array([2.0, 0.0], dtype=complex)
    covariance_power = float(
        np.vdot(linearization_trial, linearization_trial).real
    )
    affine_covariance_lower_bound = float(
        2.0
        * np.real(
            np.vdot(linearization_reference, linearization_trial)
        )
        - np.vdot(
            linearization_reference, linearization_reference
        ).real
    )
    covariance_linearization_gap = float(
        np.vdot(
            linearization_trial - linearization_reference,
            linearization_trial - linearization_reference,
        ).real
    )

    rate_system = RIS_ISAC_System(
        M=1, K=1, L=1, noise_power=1.0, seed=seed
    )
    rate_system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
    rate_system.channels["G"] = np.zeros((1, 1), dtype=complex)
    rate_system.channels["h_d"] = np.array([[1.0e-10 + 0.0j]])
    low_snr_rate = rate_system.compute_sum_rate(
        np.ones((1, 1), dtype=complex)
    )
    low_snr_rate_oracle = 1.4426950408889633e-20
    low_snr_rate_relative_error = abs(
        low_snr_rate - low_snr_rate_oracle
    ) / low_snr_rate_oracle

    balanced_system = RIS_ISAC_System(
        M=1, K=2, L=1, noise_power=1.0, seed=seed
    )
    balanced_system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
    balanced_system.channels["G"] = np.zeros((2, 1), dtype=complex)
    balanced_system.channels["h_d"] = np.full(
        (2, 1), 1.0e100 + 0.0j
    )
    balanced_W = np.full((1, 2), 1.0e100 + 0.0j)
    balanced_sinr = balanced_system.compute_sinr(
        0, balanced_W[:, 0], balanced_W[:, 1:]
    )
    balanced_sum_rate = balanced_system.compute_sum_rate(balanced_W)

    cancellation_system = RIS_ISAC_System(
        M=1, K=1, L=2, noise_power=1.0, seed=seed
    )
    cancellation_system.channels["h_d"] = np.array(
        [[1.0e280 + 0.0j]]
    )
    cancellation_system.channels["G"] = np.array(
        [[1.0e280 + 0.0j, 1.0e-60 + 0.0j]]
    )
    cancellation_system.channels["a_bs"] = np.array(
        [1.0e280 + 0.0j]
    )
    cancellation_system.channels["a_ris"] = np.array(
        [1.0e280 + 0.0j, 1.0e-60 + 0.0j]
    )
    cancellation_system.channels["H_BR"] = np.array(
        [[-1.0 + 0.0j], [1.0 + 0.0j]]
    )
    cancellation_system.set_ris_phases(np.ones(2, dtype=complex))
    cancellation_beam = np.ones(1, dtype=complex)
    cancellation_sinr = cancellation_system.compute_sinr(
        0, cancellation_beam, np.empty((1, 0), dtype=complex)
    )
    cancellation_sensing_snr = cancellation_system.compute_snr_sensing(
        cancellation_beam.reshape(1, 1)
    )

    extreme_sensing_system = RIS_ISAC_System(
        M=1, K=1, L=1, noise_power=1.0e200, seed=seed
    )
    extreme_sensing_system.channels["a_bs"] = np.array(
        [1.0e100 + 0.0j]
    )
    extreme_sensing_system.channels["a_ris"] = np.ones(1, dtype=complex)
    extreme_sensing_system.channels["H_BR"] = np.array(
        [[1.0e100 + 0.0j]]
    )
    extreme_beam = np.array([1.0e100 + 0.0j])
    extreme_triangle_oracle = stable_triangle_sensing_snr(
        extreme_sensing_system.channels["a_bs"],
        extreme_sensing_system.channels["a_ris"],
        extreme_sensing_system.channels["H_BR"],
        extreme_beam,
        extreme_sensing_system.noise_power,
    )
    _, extreme_triangle_achieved = RISPhaseOptimizer(
        extreme_sensing_system
    ).optimize_for_snr(extreme_beam.reshape(1, 1))

    system = RIS_ISAC_System(M=4, K=2, L=12, seed=seed)
    solver = AlternatingOptimizationSolver(
        system,
        problem_type="snr",
        snr_min_dB=5.0,
        max_iter=20,
        tol=1e-4,
    )
    result = solver.solve()
    metrics = solver.evaluate(result["W"], result["theta"])
    threshold = system.sinr_thresh
    sensing_threshold = 10 ** (5.0 / 10.0)
    power_history = np.asarray(result["power_history"], dtype=float)
    tolerance = 5e-4
    checks = {
        "independent_stream_power_counterexample": bool(
            opposite_stream_snr == 2.0 and equal_stream_snr == 2.0
        ),
        "affine_covariance_lower_bound_identity": bool(
            covariance_power - affine_covariance_lower_bound
            == covariance_linearization_gap
            and affine_covariance_lower_bound > 0.0
        ),
        "phase_update_nondecreasing": bool(achieved_snr >= before_snr),
        "phase_power_covariance_at_1e_minus_20_scale": bool(
            low_scale_relative_error <= 5e-13
        ),
        "phase_solution_common_scale_invariant": bool(
            phase_scale_error <= 2e-14
        ),
        "low_scale_phase_update_nondecreasing": bool(
            low_achieved_snr
            >= low_before_snr
            - 64.0
            * np.finfo(float).eps
            * max(low_before_snr, low_achieved_snr)
        ),
        "sub_epsilon_snr_rate_preserved": bool(
            low_snr_rate_relative_error <= 5e-16
        ),
        "balanced_extreme_signal_interference_ratio": bool(
            balanced_sinr == 1.0 and balanced_sum_rate == 2.0
        ),
        "cross_scale_path_cancellation_tail": bool(
            abs(cancellation_sinr - 1.0e-120) <= 2.0e-135
            and abs(cancellation_sensing_snr - 1.0e-120) <= 2.0e-135
        ),
        "single_stream_triangle_oracle_is_scale_safe": bool(
            abs(extreme_triangle_oracle - 4.0e200) / 4.0e200
            <= 3.0e-15
            and abs(
                extreme_triangle_achieved - extreme_triangle_oracle
            )
            / extreme_triangle_oracle
            <= 3.0e-15
        ),
        "unit_modulus": bool(
            np.allclose(np.abs(theta), 1.0, rtol=0.0, atol=1e-12)
        ),
        "iteration_terminated": bool(result["converged"]),
        "power_history_nonincreasing": bool(
            np.all(np.diff(power_history) <= 1e-8 * power_history[:-1])
        ),
        "power_constraint": bool(
            metrics["power_used"] <= system.P_max * (1.0 + tolerance)
        ),
        "communication_sinr_constraints": bool(
            np.all(
                metrics["sinr_per_user"]
                >= threshold * (1.0 - tolerance)
            )
        ),
        "sensing_snr_constraint": bool(
            metrics["snr_sensing"]
            >= sensing_threshold * (1.0 - tolerance)
        ),
    }
    return {
        "schema_version": 1,
        "baseline": "ris_isac_beamforming",
        "evidence_level": "educational-surrogate",
        "paper_figure_parity": False,
        "status": "pass" if all(checks.values()) else "fail",
        "claim": "educational-local-snr-feasibility-surrogate",
        "reference": {
            "doi": "10.1109/TWC.2023.3341429",
            "paper_algorithm_or_figure_parity_claimed": False,
        },
        "parameters": {
            "seed": seed,
            "M": system.M,
            "K": system.K,
            "L": system.L,
            "power_budget_watt": system.P_max,
            "noise_power_watt": system.noise_power,
            "communication_sinr_db": system.sinr_thresh_dB,
            "sensing_snr_db": 5.0,
        },
        "monotone_phase_update": {
            "snr_before": before_snr,
            "snr_after": achieved_snr,
            "global_optimality_claimed": False,
            "method": "exact-one-coordinate-ascent-with-physical-postcheck",
            "low_common_channel_scale": low_scale,
            "low_scale_snr_before": low_before_snr,
            "low_scale_snr_after": low_achieved_snr,
            "low_scale_power_covariance_oracle": low_scale_oracle,
            "low_scale_relative_error": low_scale_relative_error,
            "phase_vector_max_absolute_difference": phase_scale_error,
        },
        "low_snr_rate": {
            "sinr": 1.0e-20,
            "value": low_snr_rate,
            "log1p_oracle": low_snr_rate_oracle,
            "relative_error": low_snr_rate_relative_error,
            "relative_tolerance": 5e-16,
        },
        "scale_safe_metrics": {
            "opposite_unit_streams_sensing_snr": opposite_stream_snr,
            "equal_unit_streams_sensing_snr": equal_stream_snr,
            "balanced_signal_interference_sinr": balanced_sinr,
            "balanced_two_user_sum_rate": balanced_sum_rate,
            "cancellation_tail_oracle": 1.0e-120,
            "cancellation_tail_sinr": cancellation_sinr,
            "cancellation_tail_sensing_snr": cancellation_sensing_snr,
            "extreme_triangle_oracle": extreme_triangle_oracle,
            "extreme_triangle_achieved": extreme_triangle_achieved,
        },
        "covariance_constraint_oracle": {
            "reference_projection_real_imag": [
                [float(value.real), float(value.imag)]
                for value in linearization_reference
            ],
            "trial_projection_real_imag": [
                [float(value.real), float(value.imag)]
                for value in linearization_trial
            ],
            "covariance_power": covariance_power,
            "affine_lower_bound": affine_covariance_lower_bound,
            "squared_distance_gap": covariance_linearization_gap,
            "contains_coherent_stream_sum": False,
        },
        "solution": {
            "iterations": result["iterations"],
            "power_history_watt": power_history.tolist(),
            "power_used_watt": metrics["power_used"],
            "sinr_per_user": metrics["sinr_per_user"].tolist(),
            "sensing_snr": metrics["snr_sensing"],
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
