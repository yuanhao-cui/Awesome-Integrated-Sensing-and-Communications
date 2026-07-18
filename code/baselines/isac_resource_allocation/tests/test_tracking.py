"""Independent tests for the local constant-velocity covariance surrogate."""

import numpy as np
import pytest

from ..src.localization_qos import LocalizationQoS
from ..src.system_model import ISACSystem
from ..src.tracking_qos import TargetState, TrackingQoS


@pytest.fixture
def system():
    return ISACSystem(
        Nt=32,
        Nr=32,
        Q=3,
        K=3,
        L=1,
        fc=30e9,
        P_total=40.0,
        B_total=100e6,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def tracking_qos(system):
    return TrackingQoS(system, dt=0.1, process_noise_std=0.5)


def test_pcrb_recursive_is_finite_symmetric_and_positive(tracking_qos):
    power = np.array([10.0, 15.0, 10.0])
    bandwidth = np.array([30e6, 30e6, 25e6])
    first = tracking_qos.compute_pcrb(power, bandwidth)
    second = tracking_qos.compute_pcrb(power, bandwidth, prior_pcrb=first)
    for batch in (first, second):
        assert np.all(np.isfinite(batch))
        np.testing.assert_allclose(
            batch, np.swapaxes(batch, 1, 2), rtol=0.0, atol=1e-12
        )
        for matrix in batch:
            assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10


def test_compute_fim_default_target_uses_scalar_resources(tracking_qos):
    power = np.array([10.0, 15.0, 10.0])
    bandwidth = np.array([30e6, 30e6, 25e6])
    fim = tracking_qos.compute_fim(power, bandwidth)
    assert fim.shape == (4, 4)
    assert np.all(np.isfinite(fim))


def test_each_target_pcrb_uses_its_own_channel(tracking_qos):
    power = np.full(3, 10.0)
    bandwidth = np.array([30e6, 20e6, 25e6])
    baseline = tracking_qos.compute_pcrb(power, bandwidth)
    tracking_qos.system.beta_sensing[1] *= 1e6
    changed = tracking_qos.compute_pcrb(power, bandwidth)
    np.testing.assert_allclose(changed[0], baseline[0])
    assert np.trace(changed[1, :2, :2]) < np.trace(baseline[1, :2, :2])


def test_scalar_trace_is_position_only(tracking_qos):
    power = np.array([10.0, 15.0, 10.0])
    bandwidth = np.array([30e6, 30e6, 25e6])
    pcrb = tracking_qos.compute_pcrb(power, bandwidth)
    expected = float(np.sum(np.trace(pcrb[:, :2, :2], axis1=1, axis2=2)))
    assert tracking_qos.compute_pcrb_trace(power, bandwidth) == pytest.approx(
        expected
    )


def test_tracking_error_bound_uses_position_units_only(tracking_qos):
    power = np.array([10.0, 15.0, 10.0])
    bandwidth = np.array([30e6, 30e6, 25e6])
    position_trace = tracking_qos.compute_pcrb_position_trace(power, bandwidth)
    np.testing.assert_allclose(
        tracking_qos.compute_tracking_error_bound(power, bandwidth),
        np.sqrt(position_trace),
        rtol=1e-13,
    )


def test_update_target_states_is_seeded_prediction(tracking_qos):
    initial = [state.position.copy() for state in tracking_qos.target_states]
    tracking_qos.update_target_states()
    assert all(
        not np.allclose(state.position, before)
        for state, before in zip(tracking_qos.target_states, initial)
    )


def test_simulation_history_contains_position_trace(tracking_qos):
    power = np.array([10.0, 15.0, 10.0])
    bandwidth = np.array([30e6, 30e6, 25e6])
    pcrb_history, trace_history = tracking_qos.simulate_tracking(
        power, bandwidth, num_steps=5
    )
    assert len(pcrb_history) == len(trace_history) == 5
    expected_first = float(
        np.sum(np.trace(pcrb_history[0][:, :2, :2], axis1=1, axis2=2))
    )
    assert trace_history[0] == pytest.approx(expected_first)


def test_transition_and_process_noise_models(tracking_qos):
    transition = tracking_qos._get_transition_matrix()
    assert transition.shape == (4, 4)
    np.testing.assert_allclose(np.diag(transition), 1.0)
    process_covariance = tracking_qos._get_process_noise_cov()
    np.testing.assert_allclose(process_covariance, process_covariance.T)
    assert np.min(np.linalg.eigvalsh(process_covariance)) >= -1e-12


@pytest.mark.parametrize(
    "keyword,value",
    [("dt", 1.0e308), ("process_noise_std", 1.0e308)],
)
def test_unrepresentable_motion_covariance_is_rejected_at_construction(
    system, keyword, value
):
    with pytest.raises(ValueError, match="finite process covariance"):
        TrackingQoS(system, **{keyword: value})


def test_measurement_jacobian_and_origin_rejection(tracking_qos):
    jacobian = tracking_qos._compute_measurement_jacobian(
        np.array([10.0, 20.0, 1.0, 0.5])
    )
    assert jacobian.shape == (2, 4)
    with pytest.raises(ValueError, match="origin"):
        tracking_qos._compute_measurement_jacobian(np.zeros(4))


def test_seeded_tracking_state_and_updates_are_reproducible():
    kwargs = dict(Nt=8, Nr=8, Q=2, K=1, L=1)
    first = TrackingQoS(ISACSystem(**kwargs, rng=np.random.default_rng(9)))
    second = TrackingQoS(ISACSystem(**kwargs, rng=np.random.default_rng(9)))
    first.update_target_states()
    second.update_target_states()
    for left, right in zip(first.target_states, second.target_states):
        np.testing.assert_array_equal(left.position, right.position)
        np.testing.assert_array_equal(left.velocity, right.velocity)


def test_measurement_covariance_is_shared_with_localization(system):
    localization = LocalizationQoS(system)
    tracking = TrackingQoS(system, localization_qos=localization)
    target_index = 1
    power = 3.0
    bandwidth = 12.0e6
    angle = 0.47
    covariance = tracking._compute_measurement_covariance(
        power, bandwidth, target_index, angle
    )
    power_vector = np.zeros(system.params.Q)
    bandwidth_vector = np.ones(system.params.Q)
    angle_vector = system.target_angles.copy()
    power_vector[target_index] = power
    bandwidth_vector[target_index] = bandwidth
    angle_vector[target_index] = angle
    oracle = np.diag(
        [
            localization.compute_crb_range(
                power_vector, bandwidth_vector
            )[target_index],
            localization.compute_crb_angle(
                power_vector, bandwidth_vector, angles=angle_vector
            )[target_index],
        ]
    )
    np.testing.assert_allclose(covariance, oracle, rtol=2e-15)


def test_pcrb_uses_predicted_state_and_matches_covariance_oracle():
    system = ISACSystem(
        Nt=4,
        Nr=4,
        Q=1,
        K=1,
        L=1,
        rng=np.random.default_rng(2),
    )
    system.N0 = 1.0e-12
    system.beta_sensing = np.ones(1)
    system.rcs = np.ones(1)
    localization = LocalizationQoS(system)
    tracking = TrackingQoS(
        system,
        dt=1.0,
        process_noise_std=0.0,
        localization_qos=localization,
    )
    tracking.target_states[0] = TargetState(
        position=np.array([1.0, 0.0]), velocity=np.array([0.0, 1.0])
    )
    prior = np.eye(4)[None, :, :]
    power = np.ones(1)
    bandwidth = np.ones(1)
    actual = tracking.compute_pcrb(power, bandwidth, prior)[0]

    transition = tracking._get_transition_matrix()
    predicted_covariance = transition @ prior[0] @ transition.T
    predicted_state = transition @ np.array([1.0, 0.0, 0.0, 1.0])
    radius = np.hypot(predicted_state[0], predicted_state[1])
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
    range_information = 8.0 * np.pi**2 / (system.N0 * (3.0e8) ** 2)
    angle = np.arctan2(predicted_state[1], predicted_state[0])
    angle_information = (
        1.0
        / (system.N0 * localization.angle_noise_bandwidth_hz)
        * 4
        * (4**2 - 1)
        * np.pi**2
        * np.cos(angle) ** 2
        * localization.d_lambda**2
        / 6.0
    )
    measurement_covariance = np.diag(
        [1.0 / range_information, 1.0 / angle_information]
    )
    innovation_covariance = (
        jacobian @ predicted_covariance @ jacobian.T
        + measurement_covariance
    )
    gain = np.linalg.solve(
        innovation_covariance, jacobian @ predicted_covariance
    ).T
    residual = np.eye(4) - gain @ jacobian
    oracle = (
        residual @ predicted_covariance @ residual.T
        + gain @ measurement_covariance @ gain.T
    )
    np.testing.assert_allclose(actual, oracle, rtol=2e-13, atol=2e-13)

    old_jacobian = tracking._compute_measurement_jacobian(
        np.array([1.0, 0.0, 0.0, 1.0])
    )
    old_innovation = (
        old_jacobian @ predicted_covariance @ old_jacobian.T
        + measurement_covariance
    )
    old_gain = np.linalg.solve(
        old_innovation, old_jacobian @ predicted_covariance
    ).T
    old_residual = np.eye(4) - old_gain @ old_jacobian
    old_oracle = (
        old_residual @ predicted_covariance @ old_residual.T
        + old_gain @ measurement_covariance @ old_gain.T
    )
    assert np.linalg.norm(actual - old_oracle) > 1.0e-2


def test_zero_power_returns_prediction_exactly(tracking_qos):
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(4, 4))
    prior_single = matrix @ matrix.T + np.eye(4)
    prior = np.stack([prior_single] * tracking_qos.system.params.Q)
    power = np.zeros(tracking_qos.system.params.Q)
    bandwidth = np.ones(tracking_qos.system.params.Q)
    actual = tracking_qos.compute_pcrb(power, bandwidth, prior)
    transition = tracking_qos._get_transition_matrix()
    process_covariance = tracking_qos._get_process_noise_cov()
    oracle = np.stack(
        [transition @ item @ transition.T + process_covariance for item in prior]
    )
    np.testing.assert_allclose(actual, oracle, rtol=0.0, atol=2e-15)
