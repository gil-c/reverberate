"""Head related transfer functions, as a rigid sphere and as a measured set.

Roadmap section 7 is blunt about why this exists: "two bare microphones do not
make binaural". Two points in free air give the interaural delay from path
geometry and lose the acoustic shadow, the level difference and the pinna
filtering. What is added here is the head.

**The rigid sphere is analytic, so it is the one thing here that can be
checked rather than trusted.** Its surface pressure for a unit plane wave
arriving from ``s``, at a point whose angle to ``s`` is ``gamma``, is

    H(gamma, mu) = (1 / mu^2) sum_n (2n + 1) i^(n-1) P_n(cos gamma) / h_n'(mu)

with ``mu = k a``. The derivation is worth keeping because the sign is not
recoverable afterwards: the total field is
``sum_n i^n (2n+1) P_n [j_n(kr) + A_n h_n(kr)]``, the rigid condition
``A_n = -j_n'(mu) / h_n'(mu)`` kills the radial velocity at the surface, and
the bracket collapses through the Wronskian ``j_n h_n' - j_n' h_n = -i / mu^2``.
``h_n`` is the **second** kind here, because ``numpy.fft.rfft`` synthesises
with ``e^{+i omega t}`` and an outgoing wave is ``e^{-ikr} / r``. Taking the
first kind changes no magnitude and swaps the ears.

The same addition theorem that turns ``(2n+1) P_n(cos gamma)`` into
``sum_m Y_nm(ear) Y_nm(source)`` gives the spherical harmonic coefficients in
closed form, ``H_nm = i^(n-1) Y_nm(ear) / (mu^2 h_n'(mu))``, so the numerical
projection has something exact to be tested against.

**What the sphere is not.** It has no pinna, so it carries no elevation cue
above about 4 kHz, and no torso, so it has no shoulder reflection. It is the
tested default and the reference for the measured sets; roadmap section 7.2's
simulated heads are the eventual answer and are a different item.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import eval_legendre

from reverberate.spatial.field import spherical_hankel2
from reverberate.spatial.sh import analyse, channel_count, directions, quadrature, real_sh

__all__ = [
    "EAR_AZIMUTH_DEG",
    "HEAD_RADIUS_M",
    "HrtfSet",
    "ear_directions",
    "load_hrir_sofa",
    "measured_head",
    "project",
    "sphere_head",
    "sphere_hrtf",
    "sphere_hrtf_sh",
    "woodworth_itd_s",
]

#: Radius of the sphere standing for a head, in metres. The usual figure for an
#: average adult, and the one the Duda and Martens measurements are quoted on.
HEAD_RADIUS_M = 0.0875

#: Where the ears sit on it, in degrees of azimuth from the front. Not 90: real
#: ears sit behind the coronal plane, which is what makes the front and the
#: back distinguishable at all in a model with no pinna.
EAR_AZIMUTH_DEG = 100.0


@dataclass(frozen=True)
class HrtfSet:
    """One head, sampled on directions, as complex spectra.

    ``responses`` is ``[ear, direction, frequency]`` with the left ear first.
    ``unit_vectors`` are in the ambisonic frame, ``x`` front, ``y`` left.
    """

    responses: np.ndarray
    unit_vectors: np.ndarray
    frequency_hz: np.ndarray
    description: str
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = (2, self.unit_vectors.shape[0], self.frequency_hz.size)
        if self.responses.shape != expected:
            raise ValueError(f"responses {self.responses.shape} should be {expected}")

    def record(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "directions": int(self.unit_vectors.shape[0]),
            "frequencies": int(self.frequency_hz.size),
            "has_quadrature_weights": self.weights is not None,
        }


def ear_directions(ear_azimuth_deg: float = EAR_AZIMUTH_DEG) -> np.ndarray:
    """Unit vectors to the left and right ears, in that order.

    The left ear is at positive azimuth, which is ``+y``, which is the left.
    Swapping these two rows is the other silent failure of a binaural chain.
    """
    azimuth = np.radians(np.array([ear_azimuth_deg, -ear_azimuth_deg]))
    return directions(azimuth, np.zeros(2))


def woodworth_itd_s(
    azimuth_rad: np.ndarray,
    *,
    radius_m: float = HEAD_RADIUS_M,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """Woodworth's low frequency interaural time difference, positive when the left leads.

    ``(a / c) (theta + sin theta)`` for a source at ``theta`` from the median
    plane. Approximate by construction, and used only to say that a measured
    delay has the right sign and the right order of magnitude.
    """
    theta = np.asarray(azimuth_rad, dtype=float)
    return np.asarray(radius_m / sound_speed_m_s * (theta + np.sin(theta)), dtype=float)


def _series_order(mu_max: float) -> int:
    """Orders needed for the sphere series to converge at ``mu = k a``."""
    return int(np.ceil(mu_max + 10.0 + 3.0 * np.cbrt(max(mu_max, 1.0))))


def sphere_hrtf(
    unit_vectors: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    radius_m: float = HEAD_RADIUS_M,
    ear_azimuth_deg: float = EAR_AZIMUTH_DEG,
    sound_speed_m_s: float = 343.0,
) -> HrtfSet:
    """Surface pressure of a rigid sphere, per ear, for plane waves from ``unit_vectors``.

    Normalised so that the response tends to one at zero frequency, which is
    what a pressure receiver on a sphere small against the wavelength reads.
    """
    vectors = np.atleast_2d(np.asarray(unit_vectors, dtype=float))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    mu = 2.0 * np.pi * frequency * radius_m / sound_speed_m_s
    ears = ear_directions(ear_azimuth_deg)

    order = _series_order(float(mu.max()))
    degrees = np.arange(order + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = spherical_hankel2(degrees[None, :], mu[:, None], derivative=True)
        radial = (
            (2.0 * degrees[None, :] + 1.0)
            * (1j ** (degrees[None, :] - 1))
            / (derivative * mu[:, None] ** 2)
        )
    responses = np.empty((2, vectors.shape[0], frequency.size), dtype=complex)
    for ear in range(2):
        cos_gamma = np.clip(vectors @ ears[ear], -1.0, 1.0)
        legendre = np.asarray(
            eval_legendre(degrees[None, :], cos_gamma[:, None]), dtype=float
        )  # [direction, degree]
        responses[ear] = legendre @ radial.T
    # mu = 0 is the static limit: the sphere is transparent and the pressure is 1.
    responses[:, :, mu == 0.0] = 1.0
    return HrtfSet(
        responses=responses,
        unit_vectors=vectors,
        frequency_hz=frequency,
        description=(
            f"rigid sphere, radius {radius_m * 100:.2f} cm, ears at "
            f"{ear_azimuth_deg:.0f} degrees azimuth"
        ),
    )


def sphere_hrtf_sh(
    order: int,
    frequency_hz: np.ndarray,
    *,
    radius_m: float = HEAD_RADIUS_M,
    ear_azimuth_deg: float = EAR_AZIMUTH_DEG,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """The sphere's own coefficients in closed form, ``[ear, channel, frequency]``.

    ``H_nm = i^(n-1) Y_nm(ear) / (mu^2 h_n'(mu))``, straight from the addition
    theorem. This exists so that :func:`project` has an exact answer to be
    checked against on a head whose answer is known.
    """
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    mu = 2.0 * np.pi * frequency * radius_m / sound_speed_m_s
    ears = real_sh(order, ear_directions(ear_azimuth_deg))
    degrees = np.array([n for n in range(order + 1) for _ in range(-n, n + 1)])
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = spherical_hankel2(degrees[:, None], mu[None, :], derivative=True)
        radial = (1j ** (degrees[:, None] - 1)) / (derivative * mu[None, :] ** 2)
    out = ears[:, :, None] * radial[None, :, :]
    if np.any(mu == 0.0):
        static = np.zeros(channel_count(order))
        static[0] = 1.0
        out[:, :, mu == 0.0] = static[None, :, None]
    return np.asarray(out, dtype=complex)


def project(hrtf: HrtfSet, order: int, *, regularisation: float = 1e-8) -> np.ndarray:
    """Spherical harmonic coefficients of a sampled head, ``[ear, channel, frequency]``.

    With quadrature weights this is the exact projection. Without them, as a
    measured set on an arbitrary grid, it is a regularised least squares, which
    is the same thing when the grid happens to be a quadrature and the honest
    fallback when it is not.
    """
    if hrtf.weights is not None:
        return np.asarray(
            np.stack(
                [
                    analyse(hrtf.responses[ear].T, hrtf.unit_vectors, hrtf.weights, order).T
                    for ear in range(2)
                ]
            ),
            dtype=complex,
        )
    basis = real_sh(order, hrtf.unit_vectors)
    normal = basis.T @ basis + regularisation * np.eye(channel_count(order))
    inverse = np.linalg.solve(normal, basis.T)
    return np.asarray(np.einsum("cq,eqf->ecf", inverse, hrtf.responses), dtype=complex)


def measured_head(
    path: Path | str, sample_rate_hz: float, filter_length: int
) -> tuple[HrtfSet, dict[str, Any]]:
    """A measured head from a SOFA file, on a decoder's own frequency grid.

    The impulse responses are zero padded to ``filter_length`` and transformed,
    which puts them on ``rfftfreq(filter_length, 1 / sample_rate_hz)`` exactly.
    Padding rather than truncating: a measured response carries the propagation
    delay from the loudspeaker to the head, and cutting it would take the
    interaural delay with it.

    Returns the head and what the file says about itself. **The second is not
    optional.** A measured set travels with its licence and its attribution or
    it does not travel, and the licence of a head is not the licence of the room
    it is decoded into.
    """
    responses, unit_vectors, rate, metadata = load_hrir_sofa(path)
    if not np.isclose(rate, sample_rate_hz):
        raise ValueError(
            f"{path} was measured at {rate} Hz and the decoder works at "
            f"{sample_rate_hz} Hz; resample the file rather than the decoder"
        )
    if responses.shape[2] > filter_length:
        raise ValueError(
            f"{path} holds {responses.shape[2]} taps, longer than the "
            f"{filter_length} tap decoder; a longer filter would keep them all"
        )
    spectra = np.fft.rfft(responses, n=filter_length, axis=-1)
    return (
        HrtfSet(
            responses=np.asarray(spectra, dtype=complex),
            unit_vectors=unit_vectors,
            frequency_hz=np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz),
            description=(
                f"measured: {metadata.get('listener') or Path(path).stem}, "
                f"{metadata.get('organisation') or 'unknown source'}, "
                f"{responses.shape[1]} directions"
            ),
        ),
        metadata,
    )


def load_hrir_sofa(path: Path | str) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """A measured head from a ``SimpleFreeFieldHRIR`` file.

    Returns the impulse responses ``[ear, direction, sample]``, the directions
    as unit vectors in the ambisonic frame, the sample rate, and what the file
    says about itself, which is where the licence and the attribution live.
    A measured set travels with those or it does not travel.
    """
    import sofar

    sofa = sofar.read_sofa(str(path))
    responses = np.asarray(sofa.Data_IR, dtype=float)
    if responses.ndim != 3 or responses.shape[1] != 2:
        raise ValueError(
            f"expected [measurement, 2, sample] impulse responses, got {responses.shape}"
        )
    positions = np.asarray(sofa.SourcePosition, dtype=float)
    units = str(getattr(sofa, "SourcePosition_Units", "degree, degree, metre"))
    if "degree" in units:
        azimuth = np.radians(positions[:, 0])
        elevation = np.radians(positions[:, 1])
        unit_vectors = directions(azimuth, elevation)
    else:
        unit_vectors = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    metadata = {
        "title": str(getattr(sofa, "GLOBAL_Title", "")),
        "licence": str(getattr(sofa, "GLOBAL_License", "")),
        "author": str(getattr(sofa, "GLOBAL_Author", "")),
        "organisation": str(getattr(sofa, "GLOBAL_Organization", "")),
        "listener": str(getattr(sofa, "GLOBAL_ListenerShortName", "")),
        "measurements": int(responses.shape[0]),
    }
    return (
        np.transpose(responses, (1, 0, 2)),
        unit_vectors,
        float(np.asarray(sofa.Data_SamplingRate).ravel()[0]),
        metadata,
    )


def sphere_head(
    sample_rate_hz: float, filter_length: int, *, quadrature_degree: int = 60
) -> HrtfSet:
    """The analytic rigid sphere, sampled on a decoder's own frequency grid."""
    grid, weights = quadrature(quadrature_degree)
    frequency = np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz)
    sampled = sphere_hrtf(grid, frequency)
    return HrtfSet(sampled.responses, grid, frequency, sampled.description, weights=weights)
