#!/usr/bin/env python3
"""Emit deterministic JSON certificates for the four local teaching baselines.

These checks certify narrow, declared software invariants.  They do not claim
paper-figure parity, measured performance, or optimality.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np


SEED = 20260717
BASELINES_DIR = Path(__file__).resolve().parents[1] / "code" / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

EVIDENCE_LEVELS = {
    "csi_ratio_doppler_estimation": "educational-surrogate",
    "isac_resource_allocation": "educational-surrogate",
    "ofdm_ambiguity_function": "educational-surrogate",
    "xl_mimo_beam_training": "educational-surrogate",
}


def _certificate(
    baseline: str,
    checks: dict[str, bool],
    metrics: dict[str, object],
) -> dict[str, object]:
    """Build the common machine-readable certificate envelope."""
    return {
        "schema_version": 1,
        "baseline": baseline,
        "evidence_level": EVIDENCE_LEVELS[baseline],
        "status": "pass" if all(checks.values()) else "fail",
        "paper_figure_parity": False,
        "seed": SEED,
        "checks": checks,
        "metrics": metrics,
    }


def verify_csi_ratio() -> dict[str, object]:
    """Check exact rotation, common-offset cancellation, scale, and alias data."""
    from csi_ratio_doppler_estimation.src.mobius_estimator import (
        mobius_doppler_estimate,
    )
    from csi_ratio_doppler_estimation.src.csi_ratio import (
        compute_csi_ratio,
        compute_csi_ratio_robust,
    )
    from csi_ratio_doppler_estimation.src.signal_model import (
        csi_with_doppler,
    )

    sample_interval = 1.0e-3
    expected_hz = 50.0
    time = np.arange(128, dtype=float) * sample_interval
    h1, h2 = csi_with_doppler(
        time,
        expected_hz,
        cfo_hz=173.25,
        rng=np.random.default_rng(SEED),
    )
    ratio = h1 / h2
    result = mobius_doppler_estimate(ratio, sample_interval)
    scale_regression = [1.0e-100, 1.0e-20, 1.0, 1.0e20, 1.0e100]
    scaled_errors = []
    for magnitude in scale_regression:
        scaled = mobius_doppler_estimate(
            magnitude * np.exp(0.7j) * ratio,
            sample_interval,
        )
        scaled_errors.append(
            abs(float(scaled["rotation_frequency_hz"]) - expected_hz)
        )

    z = np.exp(1j * 2.0 * np.pi * 0.2 * np.arange(503))
    nonuniform_ratio = z / (1.0 + 0.9999 * z)
    try:
        mobius_doppler_estimate(nonuniform_ratio, 1.0)
    except ValueError:
        nonuniform_rejected = True
    else:
        nonuniform_rejected = False
    try:
        compute_csi_ratio(
            np.full(4, 1.0e308 + 0.0j),
            np.full(4, 1.0e-308 + 0.0j),
        )
    except ValueError:
        quotient_overflow_rejected = True
    else:
        quotient_overflow_rejected = False

    maximum_component = 1.7e308
    maximum_value = np.array(
        [maximum_component + 1j * maximum_component]
    )
    maximum_component_ratio = compute_csi_ratio(
        maximum_value, maximum_value
    )
    maximum_robust_ratio, maximum_robust_mask = compute_csi_ratio_robust(
        maximum_value, maximum_value
    )

    h1_zero, h2_zero = csi_with_doppler(time, expected_hz, cfo_hz=0.0)
    cancellation_error = float(np.max(np.abs(ratio - h1_zero / h2_zero)))
    frequency_error = abs(float(result["rotation_frequency_hz"]) - expected_hz)
    scaled_error = max(scaled_errors)
    checks = {
        "exact_rotation_frequency": frequency_error <= 1.0e-10,
        "common_offset_cancels": cancellation_error <= 5.0e-14,
        "nonzero_complex_scale_invariant": scaled_error <= 1.0e-10,
        "alias_metadata_is_explicit": bool(result["alias_ambiguous"])
        and float(result["alias_limit_hz"]) == 500.0,
        "physical_direction_not_inferred": result["direction"] == "unknown",
        "nonuniform_mobius_traversal_is_rejected": nonuniform_rejected,
        "unrepresentable_quotient_is_rejected": quotient_overflow_rejected,
        "maximum_finite_complex_components_divide_exactly": bool(
            np.array_equal(maximum_component_ratio, np.ones(1))
            and np.array_equal(maximum_robust_ratio, np.ones(1))
            and bool(maximum_robust_mask[0])
        ),
    }
    return _certificate(
        "csi_ratio_doppler_estimation",
        checks,
        {
            "expected_rotation_hz": expected_hz,
            "estimated_rotation_hz": result["rotation_frequency_hz"],
            "frequency_absolute_error_hz": frequency_error,
            "common_offset_cancellation_max_error": cancellation_error,
            "scaled_frequency_absolute_error_hz": scaled_error,
            "complex_scale_magnitudes": scale_regression,
            "scaled_frequency_errors_hz": scaled_errors,
            "alias_limit_hz": result["alias_limit_hz"],
            "angular_coverage_rad": result["angular_coverage_rad"],
            "weighted_phase_r_squared": result["r_squared"],
            "maximum_finite_complex_component": maximum_component,
            "maximum_component_ratio_real": float(maximum_component_ratio[0].real),
            "maximum_component_ratio_imag": float(maximum_component_ratio[0].imag),
        },
    )


def verify_resource_allocation() -> dict[str, object]:
    """Certify narrow analytic and feasibility invariants of the local surrogate."""
    from isac_resource_allocation.src.ao_solver import AOSolver
    from isac_resource_allocation.src.comm_rate import CommunicationRate
    from isac_resource_allocation.src.detection_qos import DetectionQoS
    from isac_resource_allocation.src.localization_qos import (
        LocalizationQoS,
    )
    from isac_resource_allocation.src.numerics import stable_shannon_rates
    from isac_resource_allocation.src.system_model import ISACSystem
    from isac_resource_allocation.src.tracking_qos import (
        TargetState,
        TrackingQoS,
    )

    system = ISACSystem(rng=np.random.default_rng(SEED))
    bandwidth = np.full(system.params.Q, 10.0e6)
    detector = DetectionQoS(system, Pfa=0.01)
    zero_snr_probability = detector.compute_detection_probability(
        np.zeros(system.params.Q), bandwidth
    )
    localization = LocalizationQoS(system)
    zero_power_crb = localization.compute_crb_range(
        np.zeros(system.params.Q), bandwidth
    )
    localization_power = np.array([2.0, 3.0, 4.0])
    low_bandwidth = np.full(system.params.Q, 2.0e6)
    high_bandwidth = np.full(system.params.Q, 40.0e6)
    low_bandwidth_angle = localization.compute_crb_angle(
        localization_power, low_bandwidth
    )
    high_bandwidth_angle = localization.compute_crb_angle(
        localization_power, high_bandwidth
    )
    angle_bandwidth_error = float(
        np.max(np.abs(low_bandwidth_angle - high_bandwidth_angle))
    )
    range_information, angle_information = (
        localization.compute_information_components(
            localization_power, low_bandwidth
        )
    )
    score = localization.compute_information_score(
        localization_power, low_bandwidth
    )
    score_oracle = (
        localization.w_d
        * localization.range_reference_m**2
        * range_information
        + localization.w_theta
        * localization.angle_reference_rad**2
        * angle_information
    )
    score_error = float(np.max(np.abs(score - score_oracle)))

    stable_system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=3,
        K=1,
        L=1,
        rng=np.random.default_rng(SEED),
    )
    stable_system.N0 = 1.0
    stable_system.beta_sensing = np.ones(3)
    stable_system.rcs = np.ones(3)
    extreme_false_alarm = 1.0e-100
    stable_detector = DetectionQoS(stable_system, Pfa=extreme_false_alarm)
    analytic_snr = np.array([0.0, 1.0, 1.0e12])
    analytic_probability = stable_detector.compute_detection_probability(
        analytic_snr, np.ones(3)
    )
    analytic_oracle = np.exp(
        np.log(extreme_false_alarm) / (1.0 + analytic_snr)
    )
    analytic_probability_error = float(
        np.max(np.abs(analytic_probability - analytic_oracle))
    )

    def raises_value_error(function: Callable[[], object]) -> bool:
        try:
            function()
        except ValueError:
            return True
        return False

    def unrepresentable_final_rate() -> np.ndarray:
        rate_system = ISACSystem(
            Nt=2,
            Nr=2,
            Q=1,
            K=1,
            L=0,
            rng=np.random.default_rng(SEED),
        )
        rate_system.N0 = np.nextafter(0.0, 1.0)
        rate_system.beta_comm = np.ones(1)
        return CommunicationRate(rate_system).compute_rate(
            np.full(1, 1.0e308), np.full(1, 1.0e308), "comm"
        )

    numeric_domain_rejections = {
        "noise_psd_underflow": raises_value_error(
            lambda: ISACSystem(
                Nt=2, Nr=2, Q=1, K=1, L=1, N0_dBm=-1.0e308
            )
        ),
        "noise_psd_overflow": raises_value_error(
            lambda: ISACSystem(
                Nt=2, Nr=2, Q=1, K=1, L=1, N0_dBm=1.0e308
            )
        ),
        "detection_snr_overflow": raises_value_error(
            lambda: DetectionQoS(stable_system).compute_detection_probability(
                np.full(3, 1.0e308), np.ones(3), np.full(3, 2.0)
            )
        ),
        "localization_information_overflow": raises_value_error(
            lambda: LocalizationQoS(stable_system).compute_information_components(
                np.full(3, 1.0e308), np.ones(3)
            )
        ),
        "communication_rate_overflow": raises_value_error(
            unrepresentable_final_rate
        ),
        "tracking_process_covariance_overflow": raises_value_error(
            lambda: TrackingQoS(stable_system, dt=1.0e308)
        ),
    }

    subnormal_rate_system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=0,
        P_total=1.0,
        B_total=1.0e20,
        rng=np.random.default_rng(SEED),
    )
    subnormal_rate_system.N0 = 1.0e300
    subnormal_rate_system.beta_comm = np.array([1.1154638864309574e-10])
    subnormal_final_rate = float(
        CommunicationRate(subnormal_rate_system).compute_rate(
            np.array([1.0]), np.array([1.0e20]), "comm"
        )[0]
    )
    subnormal_rate_oracle = 1.60927421724467e-310
    subnormal_rate_error_ulp = abs(
        subnormal_final_rate - subnormal_rate_oracle
    ) / np.nextafter(0.0, 1.0)
    ultra_low_snr_rate = float(
        stable_shannon_rates(
            np.array([1.0e-200]),
            np.array([1.0e300]),
            np.array([1.0e-250]),
            1.0e-150,
        )[0]
    )
    ultra_low_snr_oracle = 1.4426950408889634e-300
    ultra_low_snr_error_ulp = abs(
        ultra_low_snr_rate - ultra_low_snr_oracle
    ) / np.spacing(ultra_low_snr_oracle)
    subnormal_snr_rate = float(
        stable_shannon_rates(
            np.array([7.235316928477505e-201]),
            np.array([2.9355076889182057e256]),
            np.array([4.633086916299441e183]),
            1.6652341187627602e50,
        )[0]
    )
    subnormal_snr_oracle = 2.904204843063903e-67
    subnormal_snr_error_ulp = abs(
        subnormal_snr_rate - subnormal_snr_oracle
    ) / np.spacing(subnormal_snr_oracle)
    unit_snr_cases = (
        (1.0e200, 1.0e300, 1.0e250, 1.0e150, 9.999999999999999e299),
        (1.0e100, 1.0, 1.0e200, 1.0e300, 1.0),
    )
    unit_snr_rates = [
        float(
            stable_shannon_rates(
                np.array([power]),
                np.array([bandwidth_value]),
                np.array([gain_value]),
                noise_value,
            )[0]
        )
        for power, bandwidth_value, gain_value, noise_value, _ in unit_snr_cases
    ]
    unit_snr_oracles = [case[-1] for case in unit_snr_cases]
    unit_snr_error_ulps = [
        abs(rate - oracle) / np.spacing(oracle)
        for rate, oracle in zip(unit_snr_rates, unit_snr_oracles, strict=True)
    ]

    multi_joint_system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=2,
        rng=np.random.default_rng(SEED),
    )
    multi_joint_rates = multi_joint_system.compute_communication_rate(
        np.ones(3), np.full(3, 1.0e6)
    )
    zero_joint_system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=0,
        rng=np.random.default_rng(SEED),
    )

    tracking_system = ISACSystem(
        Nt=4,
        Nr=4,
        Q=1,
        K=1,
        L=1,
        rng=np.random.default_rng(2),
    )
    tracking_system.N0 = 1.0e-12
    tracking_system.beta_sensing = np.ones(1)
    tracking_system.rcs = np.ones(1)
    tracking_localization = LocalizationQoS(tracking_system)
    tracking = TrackingQoS(
        tracking_system,
        dt=1.0,
        process_noise_std=0.0,
        localization_qos=tracking_localization,
    )
    tracking.target_states[0] = TargetState(
        position=np.array([1.0, 0.0]), velocity=np.array([0.0, 1.0])
    )
    prior = np.eye(4)[None, :, :]
    tracking_actual = tracking.compute_pcrb(
        np.ones(1), np.ones(1), prior_pcrb=prior
    )[0]
    transition = tracking._get_transition_matrix()
    predicted_covariance = transition @ prior[0] @ transition.T
    predicted_state = transition @ np.array([1.0, 0.0, 0.0, 1.0])
    radius = float(np.hypot(predicted_state[0], predicted_state[1]))
    jacobian = np.array(
        [
            [predicted_state[0] / radius, predicted_state[1] / radius, 0.0, 0.0],
            [
                -predicted_state[1] / radius**2,
                predicted_state[0] / radius**2,
                0.0,
                0.0,
            ],
        ]
    )
    range_information_oracle = (
        8.0 * np.pi**2 / (tracking_system.N0 * (3.0e8) ** 2)
    )
    predicted_angle = float(
        np.arctan2(predicted_state[1], predicted_state[0])
    )
    angle_information_oracle = (
        1.0
        / (
            tracking_system.N0
            * tracking_localization.angle_noise_bandwidth_hz
        )
        * 4
        * (4**2 - 1)
        * np.pi**2
        * np.cos(predicted_angle) ** 2
        * tracking_localization.d_lambda**2
        / 6.0
    )
    measurement_covariance = np.diag(
        [1.0 / range_information_oracle, 1.0 / angle_information_oracle]
    )
    innovation_covariance = (
        jacobian @ predicted_covariance @ jacobian.T
        + measurement_covariance
    )
    gain = np.linalg.solve(
        innovation_covariance, jacobian @ predicted_covariance
    ).T
    residual = np.eye(4) - gain @ jacobian
    tracking_oracle = (
        residual @ predicted_covariance @ residual.T
        + gain @ measurement_covariance @ gain.T
    )
    tracking_oracle_error = float(
        np.max(np.abs(tracking_actual - tracking_oracle))
    )

    phase_system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=1,
        P_total=3.0,
        B_total=1.0,
        rng=np.random.default_rng(0),
    )
    phase_system.N0 = 1.0
    phase_system.beta_sensing = np.ones(1)
    phase_system.rcs = np.ones(1)
    phase_system.beta_comm = np.ones(1)
    phase_system.beta_isac = np.ones(1)
    phase_result = AOSolver(
        phase_system, qos_type="detection", fairness="maxmin", max_iter=4
    ).solve(Gamma_c=0.9)

    threshold = 1.0e6
    result = AOSolver(
        system,
        qos_type="detection",
        fairness="maxmin",
        max_iter=20,
    ).solve(Gamma_c=threshold)
    if result.comm_rates is None:
        raise RuntimeError("solver certificate requires communication diagnostics")

    false_alarm_error = float(np.max(np.abs(zero_snr_probability - 0.01)))
    minimum_rate = float(np.min(result.comm_rates))
    power_error = abs(float(np.sum(result.p)) - system.params.P_total)
    bandwidth_error = abs(float(np.sum(result.b)) - system.params.B_total)
    objective_history = np.asarray(result.objective_history, dtype=float)
    monotonic_error = float(
        max(0.0, -float(np.min(np.diff(objective_history))))
        if objective_history.size > 1
        else 0.0
    )
    phase_minimum_rate = float(np.min(phase_result.comm_rates))
    phase_power_error = abs(
        float(np.sum(phase_result.p)) - phase_system.params.P_total
    )
    phase_bandwidth_error = abs(
        float(np.sum(phase_result.b)) - phase_system.params.B_total
    )
    checks = {
        "zero_snr_equals_false_alarm": false_alarm_error <= 1.0e-14,
        "nonzero_scaled_chi_square_oracle": analytic_probability_error
        <= 1.0e-15
        and bool(np.all(analytic_probability > 0.0))
        and bool(np.all(np.diff(analytic_probability) > 0.0)),
        "joint_user_count_controls_channel_dimensions": (
            multi_joint_system.beta_isac.shape == (2,)
            and multi_joint_system.alpha_isac.shape == (2,)
            and multi_joint_rates.shape == (3,)
            and zero_joint_system.beta_isac.shape == (0,)
            and zero_joint_system.alpha_isac.shape == (0,)
        ),
        "extreme_finite_inputs_fail_with_explicit_domain_errors": all(
            numeric_domain_rejections.values()
        ),
        "representable_subnormal_final_rate_is_preserved": bool(
            subnormal_final_rate > 0.0 and subnormal_rate_error_ulp <= 8.0
        ),
        "ultra_low_snr_rate_is_ulp_accurate": bool(
            ultra_low_snr_rate > 0.0 and ultra_low_snr_error_ulp <= 8.0
        ),
        "subnormal_snr_double_rounding_is_ulp_accurate": bool(
            subnormal_snr_rate > 0.0 and subnormal_snr_error_ulp <= 1.0
        ),
        "unit_snr_rate_is_ulp_accurate": bool(
            max(unit_snr_error_ulps) <= 8.0
        ),
        "zero_power_has_infinite_range_crb": bool(np.all(np.isinf(zero_power_crb))),
        "angle_proxy_is_allocation_bandwidth_independent": angle_bandwidth_error
        == 0.0,
        "localization_score_is_dimensionless_by_construction": score_error
        <= 1.0e-14
        and localization.range_reference_m > 0.0
        and localization.angle_reference_rad > 0.0,
        "predicted_state_pcrb_matches_covariance_oracle": tracking_oracle_error
        <= 2.0e-13,
        "phase_one_counterexample_is_feasible": phase_minimum_rate >= 0.9
        and phase_power_error <= 1.0e-14
        and phase_bandwidth_error <= 1.0e-14,
        "best_feasible_objective_history_is_monotonic": monotonic_error == 0.0
        and bool(result.diagnostics["best_feasible_returned"]),
        "per_user_rate_threshold": minimum_rate >= threshold,
        "power_budget": power_error <= 1.0e-9,
        "bandwidth_budget": bandwidth_error <= 1.0e-5,
        "allocations_are_finite": bool(
            np.all(np.isfinite(result.p)) and np.all(np.isfinite(result.b))
        ),
    }
    return _certificate(
        "isac_resource_allocation",
        checks,
        {
            "false_alarm_probability": 0.01,
            "zero_snr_probability": zero_snr_probability.tolist(),
            "false_alarm_absolute_error": false_alarm_error,
            "extreme_false_alarm_probability": extreme_false_alarm,
            "analytic_snr": analytic_snr.tolist(),
            "analytic_detection_probability": analytic_probability.tolist(),
            "analytic_detection_oracle": analytic_oracle.tolist(),
            "analytic_detection_max_absolute_error": analytic_probability_error,
            "two_joint_user_channel_count": int(
                multi_joint_system.beta_isac.size
            ),
            "zero_joint_user_channel_count": int(
                zero_joint_system.beta_isac.size
            ),
            "numeric_domain_rejections": numeric_domain_rejections,
            "subnormal_final_rate_bit_per_second": subnormal_final_rate,
            "subnormal_final_rate_oracle_bit_per_second": subnormal_rate_oracle,
            "subnormal_final_rate_error_ulp": subnormal_rate_error_ulp,
            "ultra_low_snr_rate_bit_per_second": ultra_low_snr_rate,
            "ultra_low_snr_rate_oracle_bit_per_second": (
                ultra_low_snr_oracle
            ),
            "ultra_low_snr_rate_error_ulp": ultra_low_snr_error_ulp,
            "subnormal_snr_rate_bit_per_second": subnormal_snr_rate,
            "subnormal_snr_rate_oracle_bit_per_second": (
                subnormal_snr_oracle
            ),
            "subnormal_snr_rate_error_ulp": subnormal_snr_error_ulp,
            "unit_snr_rates_bit_per_second": unit_snr_rates,
            "unit_snr_rate_oracles_bit_per_second": unit_snr_oracles,
            "unit_snr_rate_error_ulp": unit_snr_error_ulps,
            "angle_bandwidth_invariance_absolute_error": angle_bandwidth_error,
            "dimensionless_information_score": score.tolist(),
            "dimensionless_score_oracle_max_absolute_error": score_error,
            "range_reference_m": localization.range_reference_m,
            "angle_reference_rad": localization.angle_reference_rad,
            "angle_noise_bandwidth_hz": localization.angle_noise_bandwidth_hz,
            "predicted_state_pcrb_max_absolute_error": tracking_oracle_error,
            "tracking_position_trace_square_metre": float(
                np.trace(tracking_actual[:2, :2])
            ),
            "phase_one_minimum_rate_bit_per_second": phase_minimum_rate,
            "phase_one_power_budget_absolute_error_watt": phase_power_error,
            "phase_one_bandwidth_budget_absolute_error_hz": phase_bandwidth_error,
            "phase_one_status": phase_result.diagnostics["phase_one"],
            "objective_history": objective_history.tolist(),
            "objective_monotonic_decrease": monotonic_error,
            "minimum_user_rate_bit_per_second": minimum_rate,
            "rate_threshold_bit_per_second": threshold,
            "power_budget_absolute_error_watt": power_error,
            "bandwidth_budget_absolute_error_hz": bandwidth_error,
            "iterations": result.iterations,
        },
    )


def verify_ofdm_ambiguity() -> dict[str, object]:
    """Check normalization, seed repeatability, PAPR, and range convention."""
    from ofdm_ambiguity_function.ofdm_ambiguity import (
        compute_ambiguity_function,
        compute_papr,
        compute_range_resolution,
        generate_lfm_signal,
        generate_ofdm_signal,
    )

    first = generate_ofdm_signal(
        n_subcarriers=32,
        cp_len=8,
        rng=np.random.default_rng(SEED),
    )
    second = generate_ofdm_signal(
        n_subcarriers=32,
        cp_len=8,
        rng=np.random.default_rng(SEED),
    )
    ambiguity = compute_ambiguity_function(
        first,
        np.array([-1, 0, 1]),
        np.array([0.0]),
        fs=1.0,
    )
    origin_error = abs(float(ambiguity[0, 1]) - 1.0)
    lfm = generate_lfm_signal(
        bandwidth=20.0e6,
        pulse_width=10.0e-6,
        fs=40.0e6,
    )
    lfm_papr = float(compute_papr(lfm))
    range_resolution = float(compute_range_resolution(20.0e6))
    # The teaching helper explicitly declares the conventional c = 3e8 m/s
    # approximation, so the independent arithmetic oracle uses that constant.
    expected_range = 3.0e8 / (2.0 * 20.0e6)
    scale_regression = [1.0e-200, 1.0e-160, 1.0, 1.0e160, 1.0e200]
    scale_origins = []
    scale_paprs = []
    for amplitude in scale_regression:
        scaled_signal = amplitude * np.array([1.0, 1.0j, -1.0])
        scaled_af = compute_ambiguity_function(
            scaled_signal, np.array([0]), np.array([0.0])
        )
        scale_origins.append(float(scaled_af[0, 0]))
        scale_paprs.append(float(compute_papr(scaled_signal)))
    maximum_component = 1.7e308
    maximum_component_signal = np.array(
        [
            maximum_component + 1j * maximum_component,
            -maximum_component + 1j * maximum_component,
        ]
    )
    maximum_component_af = compute_ambiguity_function(
        maximum_component_signal, np.array([0]), np.array([0.0])
    )
    maximum_component_origin = float(maximum_component_af[0, 0])
    maximum_component_papr = float(compute_papr(maximum_component_signal))
    cancellation_signal = np.array(
        [1.0e40, 1.0e-40, 1.0e-40, -1.0e40],
        dtype=np.complex128,
    )
    cancellation_power = compute_ambiguity_function(
        cancellation_signal,
        np.array([-1, 1]),
        np.array([0.0]),
    )[0]
    cancellation_power_oracle = 2.5e-321
    cancellation_power_error_ulp = float(
        np.max(np.abs(cancellation_power - cancellation_power_oracle))
        / np.nextafter(0.0, 1.0)
    )
    periodic_signal = np.array([1.0, 0.5j, -0.25, 0.75j])
    periodic_af = compute_ambiguity_function(
        periodic_signal,
        np.array([-1, 0, 1]),
        np.array([0.2, 21.2]),
        fs=3.0,
    )
    extreme_af = compute_ambiguity_function(
        periodic_signal,
        np.array([0]),
        np.array([1.0e308]),
        fs=5.0e-324,
    )
    reduced_extreme_af = compute_ambiguity_function(
        periodic_signal,
        np.array([0]),
        np.array([np.remainder(1.0e308, 5.0e-324)]),
        fs=5.0e-324,
    )
    large_bandwidth_resolution = float(compute_range_resolution(1.0e308))
    checks = {
        "ambiguity_power_origin_is_one": origin_error <= 1.0e-14,
        "ambiguity_cancellation_tail_is_preserved": bool(
            np.all(cancellation_power > 0.0)
            and cancellation_power_error_ulp <= 2.0
        ),
        "seeded_ofdm_is_bitwise_repeatable": bool(np.array_equal(first, second)),
        "ideal_lfm_has_unit_linear_papr": abs(lfm_papr - 1.0) <= 1.0e-14,
        "range_uses_two_way_c_over_2b": abs(range_resolution - expected_range)
        <= 1.0e-14,
        "normalized_metrics_are_scale_invariant": bool(
            np.allclose(scale_origins, 1.0, rtol=0.0, atol=5.0e-16)
            and np.allclose(scale_paprs, 1.0, rtol=0.0, atol=5.0e-16)
        ),
        "maximum_finite_complex_components_are_scale_safe": bool(
            maximum_component_origin == 1.0
            and maximum_component_papr == 1.0
        ),
        "doppler_is_periodic_modulo_sample_rate": bool(
            np.allclose(periodic_af[0], periodic_af[1], rtol=0.0, atol=2.0e-15)
            and np.allclose(extreme_af, reduced_extreme_af, rtol=0.0, atol=1.0e-15)
        ),
        "large_finite_bandwidth_is_scale_safe": large_bandwidth_resolution
        == 1.5e-300,
    }
    return _certificate(
        "ofdm_ambiguity_function",
        checks,
        {
            "ambiguity_origin_absolute_error": origin_error,
            "lfm_papr_linear": lfm_papr,
            "lfm_papr_db": 10.0 * np.log10(lfm_papr),
            "range_resolution_20mhz_m": range_resolution,
            "expected_range_resolution_20mhz_m": expected_range,
            "amplitude_scales": scale_regression,
            "ambiguity_origins_by_scale": scale_origins,
            "papr_by_scale": scale_paprs,
            "maximum_finite_complex_component": maximum_component,
            "maximum_component_ambiguity_origin": maximum_component_origin,
            "maximum_component_papr": maximum_component_papr,
            "cancellation_tail_power": cancellation_power.tolist(),
            "cancellation_tail_power_oracle": cancellation_power_oracle,
            "cancellation_tail_power_error_ulp": (
                cancellation_power_error_ulp
            ),
            "large_bandwidth_range_resolution_m": large_bandwidth_resolution,
        },
    )


def verify_xl_mimo() -> dict[str, object]:
    """Check Hermitian coherent gain, low-SNR stability, and seeded data."""
    import torch

    from xl_mimo_beam_training.src.utils import (
        _exact_coherent_summary,
        generate_synthetic_data,
        rate_func,
        trans_vrf,
    )

    antennas = 64
    channel = torch.ones((1, antennas), dtype=torch.complex64)
    phases = torch.zeros((1, antennas), dtype=torch.float32)
    snr = torch.ones((1, 1), dtype=torch.float32)
    coherent_rate = float(
        -rate_func(channel, phases, snr, num_antennas=antennas).item()
    )
    expected_rate = float(np.log2(antennas + 1.0))

    tiny_channel = torch.full((1, 256), 1.0e-6 + 0.0j, dtype=torch.complex64)
    tiny_rate = float(
        -rate_func(
            tiny_channel,
            torch.zeros((1, 256)),
            torch.ones((1, 1)),
            num_antennas=256,
        ).item()
    )
    expected_tiny_rate = float(np.log1p(2.56e-10) / np.log(2.0))
    cancellation_channel = torch.tensor(
        [[1.0 + 0.0j, -1.0 + 1.0e-10 + 0.0j]],
        dtype=torch.complex128,
    )
    cancellation_rate = float(
        -rate_func(
            cancellation_channel,
            torch.zeros((1, 2)),
            torch.ones((1, 1)),
            num_antennas=2,
        ).item()
    )
    expected_cancellation_rate = float(np.log1p(0.5e-20) / np.log(2.0))
    tiny_dynamic_channel = torch.tensor([[1.0e-30 + 0.0j]], dtype=torch.complex64)
    tiny_dynamic_rate = float(
        -rate_func(
            tiny_dynamic_channel,
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            num_antennas=1,
        ).item()
    )
    large_channel = torch.tensor([[1.0e20 + 0.0j]], dtype=torch.complex64)
    large_rate = float(
        -rate_func(
            large_channel,
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            num_antennas=1,
        ).item()
    )
    unit_tail_channel = torch.tensor(
        [[1.0e200 + 0.0j, -1.0e200 + 0.0j, 1.0 + 0.0j]],
        dtype=torch.complex128,
    )
    unit_tail_rate = float(
        -rate_func(
            unit_tail_channel,
            torch.zeros((1, 3), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            num_antennas=3,
        ).item()
    )
    expected_unit_tail_rate = float(np.log2(4.0 / 3.0))
    subrange_tail_channel = torch.tensor(
        [[1.0e280 + 0.0j, -1.0e280 + 0.0j, 1.0e-60 + 0.0j]],
        dtype=torch.complex128,
    )
    subrange_tail_rate = float(
        -rate_func(
            subrange_tail_channel,
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([[1.0e100]], dtype=torch.float64),
            num_antennas=3,
        ).item()
    )
    expected_subrange_tail_rate = float(np.log1p(1.0e-20 / 3.0) / np.log(2.0))
    gradient_channel = torch.tensor(
        [[1.0e280 + 1.0e280j, -1.0e280 - 1.0e280j, 1.0e-300 + 0.0j]],
        dtype=torch.complex128,
    )
    gradient_phases = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    gradient_rate_tensor = -rate_func(
        gradient_channel,
        gradient_phases,
        torch.tensor([[1.0e300]], dtype=torch.float64),
        num_antennas=3,
    )
    gradient_rate_tensor.backward()
    gradient_rate = float(gradient_rate_tensor.item())
    phase_gradient = gradient_phases.grad.detach().numpy()[0]
    expected_gradient_rate = float(np.log1p(1.0e-300 / 3.0) / np.log(2.0))
    expected_gradient_magnitude = float(
        2.0 * np.pi / (3.0 * np.log(2.0)) * 1.0e280
    )
    expected_phase_gradient = np.array(
        [expected_gradient_magnitude, -expected_gradient_magnitude, 0.0]
    )
    product_overflow_amplitude = 1.7e308
    product_overflow_channel = torch.tensor(
        [
            [
                complex(product_overflow_amplitude, product_overflow_amplitude),
                complex(-product_overflow_amplitude, -product_overflow_amplitude),
            ]
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    product_overflow_phases = torch.tensor(
        [[-0.25, -0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    product_overflow_summary = _exact_coherent_summary(
        product_overflow_channel.detach().numpy()[0],
        trans_vrf(product_overflow_phases.detach()).numpy()[0],
    )
    product_overflow_coherent_real = product_overflow_summary[1]
    product_overflow_coherent_imag = product_overflow_summary[2]
    product_overflow_rate_tensor = -rate_func(
        product_overflow_channel,
        product_overflow_phases,
        torch.ones((1, 1), dtype=torch.float64),
        num_antennas=2,
    )
    product_overflow_rate_tensor.backward()
    product_overflow_rate = float(product_overflow_rate_tensor.item())
    product_overflow_channel_gradient = (
        product_overflow_channel.grad.detach().numpy()[0]
    )
    product_overflow_phase_gradient = (
        product_overflow_phases.grad.detach().numpy()[0]
    )
    unrepresentable_gradient_channel = torch.tensor(
        [[1.0e308 + 1.0e308j, -1.0e308 - 1.0e308j, 1.0e-300 + 0.0j]],
        dtype=torch.complex128,
    )
    unrepresentable_gradient_phases = torch.zeros(
        (1, 3), dtype=torch.float64, requires_grad=True
    )
    unrepresentable_gradient_rate = -rate_func(
        unrepresentable_gradient_channel,
        unrepresentable_gradient_phases,
        torch.tensor([[1.0e300]], dtype=torch.float64),
        num_antennas=3,
    )
    try:
        unrepresentable_gradient_rate.backward()
    except FloatingPointError:
        unrepresentable_gradient_rejected = True
    else:
        unrepresentable_gradient_rejected = False

    scaled_gradient_phases = torch.zeros(
        (1, 3), dtype=torch.float64, requires_grad=True
    )
    scaled_gradient_rate = 0.1 * (
        -rate_func(
            unrepresentable_gradient_channel,
            scaled_gradient_phases,
            torch.tensor([[1.0e300]], dtype=torch.float64),
            num_antennas=3,
        )
    )
    scaled_gradient_rate.backward()
    scaled_phase_gradient = scaled_gradient_phases.grad.detach().numpy()[0]
    expected_scaled_gradient_magnitude = float(
        0.1 * 2.0 * np.pi / (3.0 * np.log(2.0)) * 1.0e308
    )
    expected_scaled_phase_gradient = np.array(
        [
            expected_scaled_gradient_magnitude,
            -expected_scaled_gradient_magnitude,
            0.0,
        ]
    )

    complex64_gradient_channel = torch.tensor(
        [[1.0e-40 + 0.0j]],
        dtype=torch.complex64,
        requires_grad=True,
    )
    complex64_gradient_rate = -rate_func(
        complex64_gradient_channel,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.tensor([[1.0e308]], dtype=torch.float64),
        num_antennas=1,
    )
    try:
        complex64_gradient_rate.backward()
    except FloatingPointError:
        complex64_gradient_rejected = True
    else:
        complex64_gradient_rejected = False

    float32_gradient_phases = torch.zeros(
        (1, 3), dtype=torch.float32, requires_grad=True
    )
    float32_gradient_rate = -rate_func(
        torch.tensor(
            [[1.0e100 + 1.0e100j, -1.0e100 - 1.0e100j, 1.0e-100 + 0.0j]],
            dtype=torch.complex128,
        ),
        float32_gradient_phases,
        torch.tensor([[1.0e100]], dtype=torch.float64),
        num_antennas=3,
    )
    try:
        float32_gradient_rate.backward()
    except FloatingPointError:
        float32_gradient_rejected = True
    else:
        float32_gradient_rejected = False

    roundable_subnormal_phases = torch.zeros(
        (1, 2), dtype=torch.float32, requires_grad=True
    )
    float32_smallest_subnormal = float(np.finfo(np.float32).smallest_subnormal)
    subnormal_local_magnitude = 0.75 * np.pi / np.log(2.0)
    subnormal_upstream_scale = (
        0.75 * float32_smallest_subnormal / subnormal_local_magnitude
    )
    roundable_subnormal_rate = subnormal_upstream_scale * (
        -rate_func(
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 1.0j]],
                dtype=torch.complex128,
            ),
            roundable_subnormal_phases,
            torch.tensor([[3.0]], dtype=torch.float64),
            num_antennas=2,
        )
    )
    roundable_subnormal_rate.backward()
    roundable_subnormal_gradient = (
        roundable_subnormal_phases.grad.detach().numpy()[0]
    )
    expected_roundable_subnormal_gradient = np.array(
        [-float32_smallest_subnormal, float32_smallest_subnormal],
        dtype=np.float32,
    )

    roundable_upper_edge_phases = torch.zeros(
        (1, 2), dtype=torch.float32, requires_grad=True
    )
    roundable_upper_edge_rate = 1.0010453452093313e38 * (
        -rate_func(
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 1.0j]],
                dtype=torch.complex128,
            ),
            roundable_upper_edge_phases,
            torch.tensor([[3.0]], dtype=torch.float64),
            num_antennas=2,
        )
    )
    roundable_upper_edge_rate.backward()
    roundable_upper_edge_gradient = (
        roundable_upper_edge_phases.grad.detach().numpy()[0]
    )
    float32_maximum = float(np.finfo(np.float32).max)
    expected_roundable_upper_edge_gradient = np.array(
        [-float32_maximum, float32_maximum],
        dtype=np.float32,
    )
    first = generate_synthetic_data(8, antennas, seed=SEED)
    second = generate_synthetic_data(8, antennas, seed=SEED)
    checks = {
        "nondefault_array_normalization": abs(coherent_rate - expected_rate)
        <= 1.0e-6,
        "tiny_positive_rate_preserved": tiny_rate > 0.0
        and abs(tiny_rate - expected_tiny_rate) <= 3.0e-15,
        "seeded_channel_data_is_repeatable": bool(
            np.array_equal(first[0], second[0])
            and np.array_equal(first[1], second[1])
        ),
        "complex128_cancellation_residual_preserved": abs(
            cancellation_rate - expected_cancellation_rate
        )
        <= 2.0e-7 * expected_cancellation_rate,
        "complex64_tiny_channel_rate_is_positive": tiny_dynamic_rate > 0.0,
        "complex64_large_channel_rate_is_finite": bool(np.isfinite(large_rate)),
        "unit_tail_survives_1e200_exact_cancellation": abs(
            unit_tail_rate - expected_unit_tail_rate
        )
        <= 2.0e-14 * expected_unit_tail_rate,
        "subrange_tail_survives_1e280_exact_cancellation": abs(
            subrange_tail_rate - expected_subrange_tail_rate
        )
        <= 2.0e-13 * expected_subrange_tail_rate,
        "final_rate_autograd_is_finite_after_extreme_cancellation": bool(
            np.all(np.isfinite(phase_gradient))
            and abs(gradient_rate - expected_gradient_rate)
            <= 3.0e-13 * expected_gradient_rate
            and np.allclose(
                phase_gradient,
                expected_phase_gradient,
                rtol=3.0e-13,
                atol=0.0,
            )
        ),
        "overflowing_exact_products_cancel_before_rounding": bool(
            product_overflow_coherent_real == 0
            and product_overflow_coherent_imag == 0
            and product_overflow_rate == 0.0
            and np.array_equal(
                product_overflow_channel_gradient,
                np.zeros(2, dtype=np.complex128),
            )
            and np.array_equal(
                product_overflow_phase_gradient,
                np.zeros(2, dtype=float),
            )
        ),
        "unrepresentable_final_gradient_is_explicit": (
            unrepresentable_gradient_rejected
        ),
        "final_gradient_scaling_precedes_range_check": bool(
            np.all(np.isfinite(scaled_phase_gradient))
            and np.allclose(
                scaled_phase_gradient,
                expected_scaled_phase_gradient,
                rtol=3.0e-13,
                atol=0.0,
            )
        ),
        "target_dtype_gradient_overflow_is_explicit": bool(
            complex64_gradient_rejected and float32_gradient_rejected
        ),
        "roundable_float32_subnormal_gradient_is_preserved": bool(
            np.array_equal(
                roundable_subnormal_gradient,
                expected_roundable_subnormal_gradient,
            )
        ),
        "roundable_float32_upper_edge_gradient_is_preserved": bool(
            np.array_equal(
                roundable_upper_edge_gradient,
                expected_roundable_upper_edge_gradient,
            )
        ),
    }
    return _certificate(
        "xl_mimo_beam_training",
        checks,
        {
            "antenna_count": antennas,
            "coherent_rate_bit_per_second_per_hz": coherent_rate,
            "expected_coherent_rate_bit_per_second_per_hz": expected_rate,
            "tiny_rate_bit_per_second_per_hz": tiny_rate,
            "expected_tiny_rate_bit_per_second_per_hz": expected_tiny_rate,
            "complex128_cancellation_rate": cancellation_rate,
            "expected_complex128_cancellation_rate": expected_cancellation_rate,
            "complex64_1e-30_rate": tiny_dynamic_rate,
            "complex64_1e20_rate": large_rate,
            "unit_tail_after_1e200_cancellation_rate": unit_tail_rate,
            "expected_unit_tail_after_1e200_cancellation_rate": (
                expected_unit_tail_rate
            ),
            "subrange_tail_after_1e280_cancellation_rate": subrange_tail_rate,
            "expected_subrange_tail_after_1e280_cancellation_rate": (
                expected_subrange_tail_rate
            ),
            "extreme_cancellation_gradient_rate": gradient_rate,
            "expected_extreme_cancellation_gradient_rate": expected_gradient_rate,
            "extreme_cancellation_phase_gradient": phase_gradient.tolist(),
            "expected_extreme_cancellation_phase_gradient": (
                expected_phase_gradient.tolist()
            ),
            "overflowing_product_cancellation_rate": product_overflow_rate,
            "overflowing_product_coherent_exact_integer_real": (
                product_overflow_coherent_real
            ),
            "overflowing_product_coherent_exact_integer_imag": (
                product_overflow_coherent_imag
            ),
            "overflowing_product_channel_gradient_real": (
                product_overflow_channel_gradient.real.tolist()
            ),
            "overflowing_product_channel_gradient_imag": (
                product_overflow_channel_gradient.imag.tolist()
            ),
            "overflowing_product_phase_gradient": (
                product_overflow_phase_gradient.tolist()
            ),
            "scaled_extreme_phase_gradient": scaled_phase_gradient.tolist(),
            "expected_scaled_extreme_phase_gradient": (
                expected_scaled_phase_gradient.tolist()
            ),
            "complex64_channel_gradient_overflow_rejected": (
                complex64_gradient_rejected
            ),
            "float32_phase_gradient_overflow_rejected": (
                float32_gradient_rejected
            ),
            "roundable_float32_subnormal_gradient": (
                roundable_subnormal_gradient.tolist()
            ),
            "expected_roundable_float32_subnormal_gradient": (
                expected_roundable_subnormal_gradient.tolist()
            ),
            "roundable_float32_upper_edge_gradient": (
                roundable_upper_edge_gradient.tolist()
            ),
            "expected_roundable_float32_upper_edge_gradient": (
                expected_roundable_upper_edge_gradient.tolist()
            ),
        },
    )


VERIFIERS: dict[str, Callable[[], dict[str, object]]] = {
    "csi_ratio_doppler_estimation": verify_csi_ratio,
    "isac_resource_allocation": verify_resource_allocation,
    "ofdm_ambiguity_function": verify_ofdm_ambiguity,
    "xl_mimo_beam_training": verify_xl_mimo,
}


def main(argv: list[str] | None = None) -> int:
    """Run one declared certificate and return nonzero on any failed check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=sorted(VERIFIERS))
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)
    certificate = VERIFIERS[args.baseline]()
    if args.json:
        print(json.dumps(certificate, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False))
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
