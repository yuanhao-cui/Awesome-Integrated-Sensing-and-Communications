"""Validated primitives for an educational Gaussian ISAC surrogate.

The communication mutual-information expression is exact for a deterministic
complex Gaussian MIMO channel and a specified input covariance.  The default
sensing map is deliberately generic.  The angle-information helper implements
a *known unit-gain, single-path* local model; it is not the nuisance-parameter
Bayesian model used for the numerical figures in the cited paper.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


Array = np.ndarray
SensingChannel = Callable[[Array], Array]
InformationMap = Callable[[Array], Array]


def _compensated_complex_matmul(left: Array, right: Array) -> Array:
    """Multiply two small reference matrices with compensated dot products.

    BLAS reductions may discard a representable low-amplitude term when much
    larger terms cancel, and the result can then depend on column order.
    ``math.fsum`` keeps non-overlapping binary64 partials for each real and
    imaginary reduction.  Individual complex products and the final sum must
    still fit binary64; violations are rejected explicitly.
    """

    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("compensated matrix factors have incompatible shapes")
    result = np.empty((a.shape[0], b.shape[1]), dtype=np.complex128)
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                try:
                    products = a[row, :] * b[:, column]
                except FloatingPointError as error:
                    raise ValueError(
                        "an individual complex matrix product exceeds binary64"
                    ) from error
            try:
                real = math.fsum(float(value) for value in products.real)
                imaginary = math.fsum(float(value) for value in products.imag)
            except OverflowError as error:
                raise ValueError("a compensated matrix sum exceeds binary64") from error
            value = complex(real, imaginary)
            if not np.isfinite(value):
                raise ValueError("a compensated matrix sum must remain finite")
            result[row, column] = value
    return result


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _noise_normalized_channel(Hc: Array, sigma_c2: float) -> Array:
    """Return ``Hc / sqrt(sigma_c2)`` without squaring ``Hc`` first.

    Forming ``Hc^H Hc / sigma_c2`` in that order can silently underflow even
    when the noise-normalized channel energy is representable.  Normalizing
    the amplitudes first avoids that failure.  Inputs whose normalized
    coefficients exceed the floating-point range are rejected explicitly.
    """

    channel = np.asarray(Hc, dtype=np.complex128)
    if channel.ndim != 2 or 0 in channel.shape:
        raise ValueError("Hc must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(channel)):
        raise ValueError("Hc must contain only finite values")
    noise_variance = _positive_finite(sigma_c2, "sigma_c2")
    noise_amplitude = float(np.sqrt(noise_variance))

    component_maximum = max(
        float(np.max(np.abs(channel.real))),
        float(np.max(np.abs(channel.imag))),
    )
    if noise_amplitude < 1:
        largest_safe_component = np.finfo(float).max * noise_amplitude
        if component_maximum > largest_safe_component:
            raise ValueError(
                "Hc / sqrt(sigma_c2) exceeds the floating-point range"
            )

    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            normalized = channel / noise_amplitude
        except FloatingPointError as error:
            raise ValueError(
                "Hc / sqrt(sigma_c2) is outside the floating-point domain"
            ) from error
    if not np.all(np.isfinite(normalized)):
        raise ValueError("noise-normalized Hc must contain only finite values")
    return normalized


def _hermitian_psd(matrix: Array, name: str) -> tuple[Array, Array]:
    """Validate a Hermitian PSD matrix and return it with its eigenvalues.

    The negative-eigenvalue tolerance is relative to the matrix scale.  It
    therefore accepts eigensolver roundoff near a nonzero spectrum without
    classifying a genuinely tiny positive eigenvalue as rank deficient.
    """

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    scale = max(float(np.linalg.norm(array, ord=2)), np.finfo(float).tiny)
    tolerance = 64 * np.finfo(float).eps * max(array.shape[0], 1) * scale
    if float(np.linalg.norm(array - array.conj().T, ord=2)) > tolerance:
        raise ValueError(f"{name} must be Hermitian")

    # Halve before adding.  ``(A + A^H) / 2`` overflows for a perfectly
    # valid Hermitian entry near ``finfo.max`` even though its average is
    # representable.
    hermitian = 0.5 * array + 0.5 * array.conj().T
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return hermitian, eigenvalues


class GaussianISACChannel:
    """Finite-dimensional Gaussian observation model with explicit RNG state.

    Parameters use the convention ``E[|Z_ij|^2] = sigma2``.  Passing a seeded
    :class:`numpy.random.Generator` makes all generated observations bitwise
    repeatable for a fixed NumPy version.
    """

    def __init__(
        self,
        Hc: Array,
        Hs_func: SensingChannel,
        sigma_c2: float,
        sigma_s2: float,
        M: int,
        Nc: int,
        Ns: int,
        T: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.M = _positive_integer(M, "M")
        self.Nc = _positive_integer(Nc, "Nc")
        self.Ns = _positive_integer(Ns, "Ns")
        self.T = _positive_integer(T, "T")
        self.sigma_c2 = _positive_finite(sigma_c2, "sigma_c2")
        self.sigma_s2 = _positive_finite(sigma_s2, "sigma_s2")

        self.Hc = np.asarray(Hc, dtype=np.complex128)
        if self.Hc.shape != (self.Nc, self.M):
            raise ValueError("Hc must have shape (Nc, M)")
        if not np.all(np.isfinite(self.Hc)):
            raise ValueError("Hc must contain only finite values")
        if not callable(Hs_func):
            raise TypeError("Hs_func must be callable")
        self.Hs_func = Hs_func

        self.rng = np.random.default_rng() if rng is None else rng
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

    def comm_channel(self) -> Array:
        """Return a copy of the communication channel."""

        return self.Hc.copy()

    def sensing_channel(self, eta: Array) -> Array:
        """Evaluate and validate the sensing channel at ``eta``."""

        channel = np.asarray(self.Hs_func(np.asarray(eta)), dtype=np.complex128)
        if channel.shape != (self.Ns, self.M):
            raise ValueError("Hs_func must return a matrix with shape (Ns, M)")
        if not np.all(np.isfinite(channel)):
            raise ValueError("Hs_func returned non-finite values")
        return channel

    def generate_noise(self, n_rows: int) -> Array:
        """Generate ``CN(0, 1)`` noise with shape ``(n_rows, T)``."""

        rows = _positive_integer(n_rows, "n_rows")
        real = self.rng.standard_normal((rows, self.T))
        imaginary = self.rng.standard_normal((rows, self.T))
        return (real + 1j * imaginary) / np.sqrt(2)

    def _waveform(self, X: Array) -> Array:
        waveform = np.asarray(X, dtype=np.complex128)
        if waveform.shape != (self.M, self.T):
            raise ValueError("X must have shape (M, T)")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("X must contain only finite values")
        return waveform

    def comm_receive(self, X: Array) -> Array:
        """Generate ``Y_c = H_c X + Z_c``."""

        waveform = self._waveform(X)
        noise = np.sqrt(self.sigma_c2) * self.generate_noise(self.Nc)
        return _compensated_complex_matmul(self.Hc, waveform) + noise

    def sense_receive(self, X: Array, eta: Array) -> Array:
        """Generate ``Y_s = H_s(eta) X + Z_s``."""

        waveform = self._waveform(X)
        channel = self.sensing_channel(eta)
        noise = np.sqrt(self.sigma_s2) * self.generate_noise(self.Ns)
        return _compensated_complex_matmul(channel, waveform) + noise


def compute_bfim(
    Rx: Array,
    T: int,
    sigma_s2: float,
    phi_func: InformationMap | None = None,
    Jp: Array | None = None,
) -> Array:
    """Evaluate ``J = (T / sigma_s2) Phi(Rx) + Jp``.

    ``Phi(Rx) = Rx`` is used only as a generic local example when no map is
    supplied.  Callers are responsible for providing the information map for
    their physical sensing model.  The prior information is additive and is
    not multiplied by the observation-length/noise factor.
    """

    interval = _positive_integer(T, "T")
    noise_variance = _positive_finite(sigma_s2, "sigma_s2")
    covariance, _ = _hermitian_psd(Rx, "Rx")

    mapped = covariance if phi_func is None else phi_func(covariance)
    information, _ = _hermitian_psd(mapped, "Phi")

    information_scale = max(
        float(np.max(np.abs(information.real))),
        float(np.max(np.abs(information.imag))),
    )
    if information_scale == 0.0:
        bfim = np.zeros_like(information)
    else:
        # Prefer ordinary arithmetic in a safe order so common exact values
        # retain their precision.  Try both associations because either
        # ``scale * T`` or ``scale / sigma`` may be the unsafe intermediate.
        output_scale: float | None = None
        try:
            interval_float = float(interval)
        except OverflowError as error:
            raise ValueError("T is outside the floating-point scaling domain") from error
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            candidates = (
                (information_scale * interval_float) / noise_variance,
                (information_scale / noise_variance) * interval_float,
            )
        for candidate in candidates:
            if np.isfinite(candidate) and candidate > 0.0:
                output_scale = float(candidate)
                break

        # If both ordinary associations failed, use logarithms to distinguish
        # a genuinely out-of-range BFIM from an avoidable intermediate error.
        try:
            log_output_scale = (
                np.log(information_scale)
                + np.log(interval_float)
                - np.log(noise_variance)
            )
        except (OverflowError, ValueError) as error:
            raise ValueError("T is outside the floating-point scaling domain") from error
        max_log = float(np.log(np.finfo(float).max))
        min_log = float(np.log(np.nextafter(0.0, 1.0)))
        if output_scale is None:
            if log_output_scale > max_log:
                raise ValueError("the observation BFIM exceeds the floating-point range")
            if log_output_scale < min_log:
                raise ValueError("the observation BFIM is below the floating-point range")
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                try:
                    output_scale = float(np.exp(log_output_scale))
                except FloatingPointError as error:
                    raise ValueError(
                        "the observation BFIM is outside the floating-point range"
                    ) from error
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                normalized_information = (
                    information.real / information_scale
                    + 1j * (information.imag / information_scale)
                )
                bfim = normalized_information * output_scale
            except FloatingPointError as error:
                raise ValueError(
                    "the observation BFIM is outside the floating-point range"
                ) from error
    if Jp is not None:
        prior, _ = _hermitian_psd(Jp, "Jp")
        if prior.shape != information.shape:
            raise ValueError("Jp must have the same shape as Phi")
        with np.errstate(over="raise", invalid="raise"):
            try:
                bfim = bfim + prior
            except FloatingPointError as error:
                raise ValueError("observation and prior BFIM sum exceeds the range") from error
    if not np.all(np.isfinite(bfim)):
        raise ValueError("BFIM must remain finite")
    return 0.5 * bfim + 0.5 * bfim.conj().T


def compute_crb(
    Rx: Array | None = None,
    T: int = 1,
    sigma_s2: float = 1.0,
    phi_func: InformationMap | None = None,
    Jp: Array | None = None,
    bfim: Array | None = None,
) -> float:
    """Return ``tr(J^-1)`` or infinity for a singular valid BFIM.

    A strictly positive eigenvalue is always inverted, irrespective of its
    absolute magnitude.  This is important for weak yet identifiable models.
    """

    if bfim is None:
        if Rx is None:
            raise ValueError("Rx is required when bfim is not supplied")
        information = compute_bfim(Rx, T, sigma_s2, phi_func, Jp)
    else:
        information = np.asarray(bfim, dtype=np.complex128)

    _, eigenvalues = _hermitian_psd(information, "bfim")
    if np.any(eigenvalues <= 0):
        return float("inf")
    reciprocal_limit = 1.0 / np.finfo(float).max
    if np.any(eigenvalues < reciprocal_limit):
        return float("inf")

    reciprocals = 1.0 / eigenvalues
    largest = float(np.max(reciprocals))
    scaled_sum = float(np.sum(reciprocals / largest))
    if largest > np.finfo(float).max / scaled_sum:
        return float("inf")
    return largest * scaled_sum


def compute_rate(Rx: Array, Hc: Array, sigma_c2: float) -> float:
    """Evaluate Gaussian MIMO mutual information in nats/channel use.

    The returned value is

    ``log det(I + Hc Rx Hc^H / sigma_c2)``.
    """

    covariance, _ = _hermitian_psd(Rx, "Rx")
    normalized_channel = _noise_normalized_channel(Hc, sigma_c2)
    if normalized_channel.shape[1] != covariance.shape[0]:
        raise ValueError("Hc and Rx have incompatible transmit dimensions")

    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
    covariance_eigenvalues = np.maximum(covariance_eigenvalues, 0.0)
    square_root = np.sqrt(covariance_eigenvalues)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            effective_channel = _compensated_complex_matmul(
                normalized_channel,
                covariance_eigenvectors,
            ) * square_root[np.newaxis, :]
        except FloatingPointError as error:
            raise ValueError(
                "the noise-normalized effective channel exceeds the "
                "floating-point range"
            ) from error
    if not np.all(np.isfinite(effective_channel)):
        raise ValueError(
            "the noise-normalized effective channel must be finite"
        )

    singular_values = np.linalg.svd(effective_channel, compute_uv=False)
    square_limit = float(np.sqrt(np.finfo(float).max))
    ordinary = singular_values <= square_limit
    terms = np.empty_like(singular_values)
    with np.errstate(under="ignore"):
        terms[ordinary] = np.log1p(singular_values[ordinary] ** 2)
        inverse = 1.0 / singular_values[~ordinary]
        terms[~ordinary] = (
            2 * np.log(singular_values[~ordinary])
            + np.log1p(inverse**2)
        )
    return float(np.sum(terms))


def compute_rate_per_symbol(X: Array, Hc: Array, sigma_c2: float) -> float:
    """Evaluate the covariance rate induced by a finite waveform matrix."""

    waveform = np.asarray(X, dtype=np.complex128)
    if waveform.ndim != 2 or 0 in waveform.shape:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("X must contain only finite values")
    interval = waveform.shape[1]
    normalized_channel = _noise_normalized_channel(Hc, sigma_c2)
    if normalized_channel.shape[1] != waveform.shape[0]:
        raise ValueError("Hc and X have incompatible transmit dimensions")

    # Evaluate through H X / sqrt(T), whose singular values give the same
    # log-determinant as H (X X^H / T) H^H.  Forming X X^H first needlessly
    # overflows for reciprocal channel/waveform scales whose received product
    # is perfectly representable.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            effective_channel = _compensated_complex_matmul(
                normalized_channel,
                waveform,
            ) / np.sqrt(float(interval))
        except FloatingPointError as error:
            raise ValueError(
                "the noise-normalized received waveform exceeds the "
                "floating-point range"
            ) from error
    if not np.all(np.isfinite(effective_channel)):
        raise ValueError("the noise-normalized received waveform must be finite")

    singular_values = np.linalg.svd(effective_channel, compute_uv=False)
    square_limit = float(np.sqrt(np.finfo(float).max))
    ordinary = singular_values <= square_limit
    terms = np.empty_like(singular_values)
    with np.errstate(under="ignore"):
        terms[ordinary] = np.log1p(singular_values[ordinary] ** 2)
        inverse = 1.0 / singular_values[~ordinary]
        terms[~ordinary] = (
            2 * np.log(singular_values[~ordinary])
            + np.log1p(inverse**2)
        )
    return float(np.sum(terms))


def make_uniform_linear_array(M: int, d: float = 0.5) -> Callable[[float], Array]:
    """Return an unnormalized broadside ULA steering-vector function.

    Angles are in radians and spacing ``d`` is in wavelengths.  The convention
    is ``exp(j 2 pi d m sin(theta))`` for indices ``m = 0, ..., M-1``.
    """

    antennas = _positive_integer(M, "M")
    spacing = float(d)
    if not np.isfinite(spacing) or spacing < 0:
        raise ValueError("d must be non-negative and finite")
    positions = np.arange(antennas, dtype=float) * spacing

    def steering_vector(theta: float) -> Array:
        angle = float(theta)
        if not np.isfinite(angle):
            raise ValueError("theta must be finite")
        values = np.exp(1j * 2 * np.pi * positions * np.sin(angle))
        return values.reshape(-1, 1)

    return steering_vector


def angle_to_channel(
    theta: float,
    M: int,
    N: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
) -> Array:
    """Return the unit-gain single-path channel ``a_rx a_tx^H``."""

    transmit = make_uniform_linear_array(M, d_tx)(theta)
    receive = make_uniform_linear_array(N, d_rx)(theta)
    return receive @ transmit.conj().T


def angle_to_hfunc(
    M: int,
    Ns: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
) -> SensingChannel:
    """Build a one-parameter, known-gain single-path sensing channel."""

    transmit_antennas = _positive_integer(M, "M")
    receive_antennas = _positive_integer(Ns, "Ns")

    def hfunc(eta: Array) -> Array:
        parameter = np.asarray(eta)
        if parameter.size != 1:
            raise ValueError("eta must contain exactly one angle")
        theta = float(parameter.reshape(-1)[0])
        return angle_to_channel(
            theta,
            transmit_antennas,
            receive_antennas,
            d_tx,
            d_rx,
        )

    return hfunc


def compute_phi_angle(
    Rx: Array,
    T: int,
    theta: float,
    M: int,
    Ns: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
) -> Array:
    """Return a known-unit-gain scalar angle-information map.

    For ``H(theta) = a_rx(theta) a_tx(theta)^H`` and circular complex
    Gaussian noise, this helper returns

    ``Phi(Rx) = 2 Re tr((dH/dtheta)^H (dH/dtheta) Rx)``.

    ``T`` is validated but not applied here because :func:`compute_bfim`
    applies the coherent-interval factor exactly once.  Unknown complex gain,
    prior averaging, and nuisance-parameter elimination are outside this
    helper and outside the accompanying numerical certificate.
    """

    _positive_integer(T, "T")
    transmit_antennas = _positive_integer(M, "M")
    receive_antennas = _positive_integer(Ns, "Ns")
    covariance, _ = _hermitian_psd(Rx, "Rx")
    if covariance.shape != (transmit_antennas, transmit_antennas):
        raise ValueError("Rx must have shape (M, M)")

    angle = float(theta)
    tx_spacing = float(d_tx)
    rx_spacing = float(d_rx)
    if not all(np.isfinite(value) for value in (angle, tx_spacing, rx_spacing)):
        raise ValueError("theta and array spacings must be finite")
    if tx_spacing < 0 or rx_spacing < 0:
        raise ValueError("array spacings must be non-negative")

    try:
        with np.errstate(over="raise", invalid="raise"):
            positions_tx = np.arange(transmit_antennas) * tx_spacing
            positions_rx = np.arange(receive_antennas) * rx_spacing
            a_tx = np.exp(1j * 2 * np.pi * positions_tx * np.sin(angle))
            a_rx = np.exp(1j * 2 * np.pi * positions_rx * np.sin(angle))
            da_tx = 1j * 2 * np.pi * positions_tx * np.cos(angle) * a_tx
            da_rx = 1j * 2 * np.pi * positions_rx * np.cos(angle) * a_rx
    except FloatingPointError as error:
        raise ValueError(
            "array geometry is outside the finite angle-information domain"
        ) from error

    derivative = np.outer(da_rx, a_tx.conj()) + np.outer(
        a_rx,
        da_tx.conj(),
    )
    weight = derivative.conj().T @ derivative
    information = 2 * float(np.real(np.trace(weight @ covariance)))
    if information < 0:
        scale = float(np.linalg.norm(weight, ord=2) * np.trace(covariance).real)
        if information < -64 * np.finfo(float).eps * max(scale, 1.0):
            raise RuntimeError("computed angle information is unexpectedly negative")
        information = 0.0
    return np.array([[information]], dtype=np.complex128)
