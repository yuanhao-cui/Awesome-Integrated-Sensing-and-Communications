"""
Integration tests for ISAC Resource Allocation framework.
"""

from decimal import Decimal, localcontext
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from ..src.system_model import ISACSystem
from ..src.detection_qos import DetectionQoS
from ..src.localization_qos import LocalizationQoS
from ..src.numerics import stable_shannon_rates
from ..src.tracking_qos import TrackingQoS
from ..src.comm_rate import CommunicationRate
from ..src.ao_solver import AOSolver
from ..src.fairness import FairnessMetrics


@pytest.fixture
def system():
    """Create ISAC system for integration testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


def test_system_model_integration(system):
    """Test system model components work together."""
    # Test channel gains are positive
    assert np.all(system.beta_sensing > 0), "Sensing channel gains should be positive"
    assert np.all(system.beta_comm > 0), "Communication channel gains should be positive"
    assert np.all(system.beta_isac > 0), "ISAC channel gains should be positive"

    # Test RCS values
    assert np.all(system.rcs > 0), "RCS values should be positive"

    # Test noise power
    assert system.N0 > 0, "Noise power should be positive"


def test_qos_modules_integration(system):
    """Test that all QoS modules work with the same system."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    # Detection QoS
    detection = DetectionQoS(system)
    P_D = detection.compute_detection_probability(p, b)
    assert np.all(P_D >= 0) and np.all(P_D <= 1), "Detection probabilities should be valid"

    # Localization QoS
    localization = LocalizationQoS(system)
    rho = localization.compute_crb_combined(p, b)
    assert np.all(rho > 0), "Localization metrics should be positive"

    # Tracking QoS
    tracking = TrackingQoS(system)
    pcrb = tracking.compute_pcrb(p, b)
    assert pcrb.shape == (system.params.Q, 4, 4), "PCRB should have correct shape"


def test_communication_rate_integration(system):
    """Test communication rate with system model."""
    comm = CommunicationRate(system)

    p_comm = np.array([10.0, 15.0, 10.0])
    b_comm = np.array([30e6, 30e6, 25e6])

    rates = comm.compute_rate(p_comm, b_comm, 'comm')
    assert np.all(rates >= 0), "Communication rates should be non-negative"

    # Test rate constraints
    satisfied, actual_rates = comm.check_rate_constraints(p_comm, b_comm, Gamma_c=1.0)
    assert len(actual_rates) == len(p_comm), "Should have rate for each user"


def test_ao_solver_full_integration(system):
    """Test AO solver with all components."""
    solver = AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=10)

    result = solver.solve(Gamma_c=1.0)

    # Validate result
    assert result.p is not None, "Power allocation should exist"
    assert result.b is not None, "Bandwidth allocation should exist"

    # Check budgets
    assert np.isclose(np.sum(result.p), system.params.P_total, rtol=1e-3)
    assert np.isclose(np.sum(result.b), system.params.B_total, rtol=1e-3)

    # Check non-negativity
    assert np.all(result.p >= 0), "Power allocation should be non-negative"
    assert np.all(result.b >= 0), "Bandwidth allocation should be non-negative"


def test_fairness_metrics_integration(system):
    """Test fairness metrics with system results."""
    fairness = FairnessMetrics()

    values = np.array([0.8, 0.9, 0.7])

    # Test all fairness metrics
    jfi = fairness.compute_jain_fairness_index(values)
    assert jfi >= 1/len(values) and jfi <= 1, "Jain fairness index should be in [1/N, 1]"

    gini = fairness.compute_gini_coefficient(values)
    assert gini >= 0 and gini <= 1, "Gini coefficient should be in [0, 1]"

    mmr = fairness.compute_min_max_ratio(values)
    assert mmr >= 0 and mmr <= 1, "Min/max ratio should be in [0, 1]"


def test_multiple_qos_integration(system):
    """Test solving for all QoS types."""
    solver = AOSolver(system, max_iter=5)

    results = solver.solve_multiple_qos(Gamma_c=1.0)

    # All results should satisfy constraints
    for qos_type, result in results.items():
        assert np.isclose(np.sum(result.p), system.params.P_total, rtol=1e-3), \
            f"Power budget violated for {qos_type}"
        assert np.isclose(np.sum(result.b), system.params.B_total, rtol=1e-3), \
            f"Bandwidth budget violated for {qos_type}"


def test_system_validation(system):
    """Test system validation functions."""
    M = system.total_objects

    # Valid allocation
    p_valid = np.ones(M) * system.params.P_total / M
    b_valid = np.ones(M) * system.params.B_total / M

    assert system.validate_allocations(p_valid, b_valid), \
        "Valid allocation should pass validation"

    # Invalid allocation (negative power)
    p_invalid = p_valid.copy()
    p_invalid[0] = -1.0

    assert not system.validate_allocations(p_invalid, b_valid), \
        "Negative power should fail validation"

    # Invalid allocation (budget violation)
    p_over = p_valid * 2
    assert not system.validate_allocations(p_over, b_valid), \
        "Power budget violation should fail validation"

    assert not system.validate_allocations(
        np.array([system.params.P_total]),
        np.array([system.params.B_total]),
    ), "Wrong-length allocations must fail even when their sums match"


def test_system_communication_rate_covers_comm_and_isac_users(system):
    """The convenience API returns K+L rates with the matching gain vectors."""
    count = system.params.K + system.params.L
    rates = system.compute_communication_rate(
        np.ones(count),
        np.full(count, 1e6),
    )
    assert rates.shape == (count,)
    assert np.all(np.isfinite(rates))


@pytest.mark.parametrize("joint_users", [0, 2])
def test_joint_user_count_controls_channel_and_solver_dimensions(joint_users):
    """L=0 and L>1 must not silently retain a one-user channel vector."""
    system = ISACSystem(
        Nt=4,
        Nr=4,
        Q=2,
        K=1,
        L=joint_users,
        P_total=10.0,
        B_total=10.0e6,
        rng=np.random.default_rng(8),
    )
    assert system.beta_isac.shape == (joint_users,)
    assert system.alpha_isac.shape == (joint_users,)
    assert system.isac_positions.shape == (joint_users,)
    result = AOSolver(system, max_iter=2).solve(Gamma_c=0.0)
    assert result.p.shape == (system.params.M,)
    assert result.b.shape == (system.params.M,)
    assert result.comm_rates.shape == (system.params.K + joint_users,)


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"Nt": 0}, ValueError),
        ({"Q": 0}, ValueError),
        ({"K": 0, "L": 0}, ValueError),
        ({"L": -1}, ValueError),
        ({"B_total": np.inf}, ValueError),
        ({"Nt": 4.5}, TypeError),
    ],
)
def test_system_rejects_invalid_dimensions_and_budgets(overrides, error_type):
    parameters = {"Nt": 4, "Nr": 4, "Q": 1, "K": 1, "L": 1}
    parameters.update(overrides)
    with pytest.raises(error_type):
        ISACSystem(**parameters)


@pytest.mark.parametrize("noise_psd_dbm", [-1.0e308, 1.0e308])
def test_system_rejects_unrepresentable_noise_psd_without_warning(noise_psd_dbm):
    with pytest.raises(ValueError, match="finite positive noise PSD"):
        ISACSystem(Nt=4, Nr=4, Q=1, K=1, L=1, N0_dBm=noise_psd_dbm)


def test_system_rate_preserves_representable_extremes_and_rejects_final_overflow():
    system = ISACSystem(Nt=4, Nr=4, Q=1, K=1, L=1)
    system.N0 = 1.0
    system.beta_comm = np.ones(1)
    system.beta_isac = np.ones(1)
    representable = system.compute_communication_rate(
        np.full(2, 1.0e308), np.full(2, 1.0e-300)
    )
    log_snr = np.log(1.0e308) - np.log(1.0e-300)
    oracle = 1.0e-300 * log_snr / np.log(2.0)
    np.testing.assert_allclose(representable, oracle, rtol=2e-15, atol=0.0)

    system.N0 = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="outside the finite numerical domain"):
        system.compute_communication_rate(
            np.full(2, 1.0e308), np.full(2, 1.0e308)
        )
    with pytest.raises(ValueError, match="outside the finite numerical domain"):
        CommunicationRate(system).compute_rate(
            np.array([1.0e308]), np.array([1.0e308]), "comm"
        )


def test_communication_helpers_preserve_tiny_rates_and_require_optional_pairs(system):
    comm = CommunicationRate(system)
    power = np.full(system.params.K, 1e-20)
    bandwidth = np.full(system.params.K, 1e6)
    spectral_efficiency = comm.compute_spectral_efficiency(power, bandwidth)
    assert np.all(spectral_efficiency > 0)
    np.testing.assert_allclose(
        spectral_efficiency,
        comm.compute_rate(power, bandwidth, "comm") / bandwidth,
    )
    with pytest.raises(ValueError, match="provided together"):
        comm.compute_sum_rate(power, bandwidth, p_isac=np.ones(system.params.L))


def test_communication_rate_preserves_subnormal_result_after_bandwidth_cancellation():
    system = ISACSystem(
        Nt=2,
        Nr=2,
        Q=1,
        K=1,
        L=0,
        P_total=1.0,
        B_total=1.0e20,
        rng=np.random.default_rng(1),
    )
    system.N0 = 1.0e300
    system.beta_comm = np.array([1.1154638864309574e-10])
    power = np.array([1.0])
    bandwidth = np.array([1.0e20])
    oracle = 1.60927421724467e-310
    helper_rate = CommunicationRate(system).compute_rate(
        power, bandwidth, "comm"
    )[0]
    system_rate = system.compute_communication_rate(power, bandwidth)[0]
    assert helper_rate > 0.0
    assert abs(helper_rate - oracle) <= 8 * np.nextafter(0.0, 1.0)
    assert system_rate == helper_rate


def test_ultra_low_snr_rate_avoids_log_domain_precision_loss():
    system = ISACSystem(Nt=2, Nr=2, Q=1, K=1, L=0)
    system.N0 = 1.0e-150
    system.beta_comm = np.array([1.0e-250])
    rate = CommunicationRate(system).compute_rate(
        np.array([1.0e-200]),
        np.array([1.0e300]),
        "comm",
    )[0]
    oracle = 1.4426950408889634e-300
    assert rate > 0.0
    assert abs(rate - oracle) <= 8 * np.spacing(oracle)


def test_subnormal_snr_rate_avoids_double_rounding_against_decimal_oracle():
    power = 7.235316928477505e-201
    bandwidth = 2.9355076889182057e256
    gain = 4.633086916299441e183
    noise_psd = 1.6652341187627602e50
    rate = stable_shannon_rates(
        np.array([power]),
        np.array([bandwidth]),
        np.array([gain]),
        noise_psd,
    )[0]

    # Decimal.from_float preserves the exact binary64 inputs.  Five hundred
    # digits are enough to resolve log1p at this roughly 7e-324 exact SNR.
    with localcontext() as context:
        context.prec = 500
        decimal_power = Decimal.from_float(power)
        decimal_bandwidth = Decimal.from_float(bandwidth)
        decimal_gain = Decimal.from_float(gain)
        decimal_noise = Decimal.from_float(noise_psd)
        exact_snr = (
            decimal_power
            * decimal_gain
            / (decimal_noise * decimal_bandwidth)
        )
        decimal_rate = (
            decimal_bandwidth
            * (Decimal(1) + exact_snr).ln()
            / Decimal(2).ln()
        )
    oracle = float(decimal_rate)
    assert oracle == 2.904204843063903e-67
    assert abs(rate - oracle) <= np.spacing(oracle)


@pytest.mark.parametrize(
    "power,bandwidth,gain,noise_psd,oracle",
    [
        (1.0e200, 1.0e300, 1.0e250, 1.0e150, 9.999999999999999e299),
        (1.0e100, 1.0, 1.0e200, 1.0e300, 1.0),
    ],
)
def test_unit_snr_rate_avoids_large_log_cancellation(
    power, bandwidth, gain, noise_psd, oracle
):
    rate = stable_shannon_rates(
        np.array([power]),
        np.array([bandwidth]),
        np.array([gain]),
        noise_psd,
    )[0]
    assert abs(rate - oracle) <= 8 * np.spacing(oracle)


def test_channel_matrix_generation(system):
    """Test channel matrix generation."""
    h_sensing = system.get_channel_matrix(0, 'sensing')
    h_comm = system.get_channel_matrix(0, 'comm')
    h_isac = system.get_channel_matrix(0, 'isac')

    # Should have correct shape
    assert h_sensing.shape == (system.params.Nt,), \
        f"Sensing channel should have shape (Nt,), got {h_sensing.shape}"
    assert h_comm.shape == (system.params.Nt,), \
        f"Comm channel should have shape (Nt,), got {h_comm.shape}"
    assert h_isac.shape == (system.params.Nt,), \
        f"ISAC channel should have shape (Nt,), got {h_isac.shape}"

    # Should have non-zero norm
    assert np.linalg.norm(h_sensing) > 0, "Sensing channel should have non-zero norm"
    assert np.linalg.norm(h_comm) > 0, "Comm channel should have non-zero norm"
    assert np.linalg.norm(h_isac) > 0, "ISAC channel should have non-zero norm"


def test_end_to_end_workflow(system):
    """Test complete end-to-end workflow."""
    # 1. Create system
    assert system.params.M == 7, "System should have 7 total objects (Q+K+L=3+3+1)"

    # 2. Create QoS modules
    detection = DetectionQoS(system)
    localization = LocalizationQoS(system)
    TrackingQoS(system)
    CommunicationRate(system)

    # 3. Create solver
    solver = AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=5)

    # 4. Solve
    result = solver.solve(Gamma_c=1.0)

    # 5. Validate
    assert result.p is not None and result.b is not None

    # 6. Compute final metrics
    Q = system.params.Q
    p_sensing = result.p[:Q]
    b_sensing = result.b[:Q]

    P_D = detection.compute_detection_probability(p_sensing, b_sensing)
    rho = localization.compute_crb_combined(p_sensing, b_sensing)

    assert np.all(P_D >= 0) and np.all(P_D <= 1)
    assert np.all(rho > 0)


def test_config_loading():
    """Test configuration file loading."""
    import yaml

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate config structure
        assert 'system' in config, "Config should have system section"
        assert 'solver' in config, "Config should have solver section"

        # Check system parameters
        sys_config = config['system']
        assert 'Nt' in sys_config and sys_config['Nt'] > 0
        assert 'Nr' in sys_config and sys_config['Nr'] > 0
        assert 'Q' in sys_config and sys_config['Q'] > 0
        assert 'K' in sys_config and sys_config['K'] > 0
        assert 'L' in sys_config and sys_config['L'] > 0


def test_performance_with_large_system():
    """Test performance with larger system."""
    rng = np.random.default_rng(42)
    large_system = ISACSystem(Nt=64, Nr=64, Q=5, K=5, L=1, fc=30e9,
                             P_total=100.0, B_total=200e6, rng=rng)

    solver = AOSolver(large_system, max_iter=3)

    # Should complete without errors
    result = solver.solve(Gamma_c=2.0)

    assert result.p is not None
    assert result.b is not None
    assert np.isclose(np.sum(result.p), large_system.params.P_total, rtol=1e-3)
