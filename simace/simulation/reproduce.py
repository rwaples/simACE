"""Offspring A, C and E components: founders, Mendelian noise, and C transmission."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from simace.simulation.pairing_kernels import _midparent
from simace.simulation.params import c_covariance
from simace.simulation.pedigree_buffers import _A1, _A2, _C1, _C2
from simace.simulation.records import CClip

if TYPE_CHECKING:
    from simace.simulation.params import CTransmission


logger = logging.getLogger(__name__)


def generate_correlated_components(
    rng: np.random.Generator, n: int, sd1: float, sd2: float, correlation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated normal variables via multivariate normal.

    Args:
        rng: numpy random generator
        n: number of samples
        sd1: standard deviation for component 1
        sd2: standard deviation for component 2
        correlation: correlation between components

    Returns:
        (comp1, comp2): tuple of arrays, each shape (n,)

    Raises:
        ValueError: if sd1 or sd2 is negative, or correlation is outside [-1, 1]
    """
    if sd1 < 0 or sd2 < 0:
        raise ValueError(f"Standard deviations must be non-negative, got sd1={sd1}, sd2={sd2}")
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    cov = [
        [sd1**2, correlation * sd1 * sd2],
        [correlation * sd1 * sd2, sd2**2],
    ]
    samples = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    return samples[:, 0], samples[:, 1]


def generate_mendelian_noise(
    rng: np.random.Generator,
    n: int,
    sd_A1: float,
    sd_A2: float,
    rA: float,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated Mendelian sampling noise for two traits.

    Under the infinitesimal model, the Mendelian sampling variance is
    0.5 * Var(A) for each trait, so sd_noise = sd_A / sqrt(2)
    (Bulmer, 1971, Am. Nat., 105, 201-211). With parental inbreeding it is
    0.5 * Var(A) * (1 - F_bar): ``scale`` multiplies each offspring's noise
    in both traits by ``sqrt(1 - F_bar)`` after the draw, so it consumes the
    same random numbers (ADR 0033).

    Args:
        rng: numpy random generator
        n: number of offspring
        sd_A1: standard deviation of A1 (sqrt of A1 variance)
        sd_A2: standard deviation of A2 (sqrt of A2 variance)
        rA: genetic correlation between traits
        scale: per-offspring factor, shape (n,), or None for none

    Returns:
        (noise1, noise2): tuple of arrays, each shape (n,)
    """
    sd_noise1 = sd_A1 / np.sqrt(2)
    sd_noise2 = sd_A2 / np.sqrt(2)
    noise1, noise2 = generate_correlated_components(rng, n, sd_noise1, sd_noise2, rA)
    if scale is not None:
        noise1, noise2 = noise1 * scale, noise2 * scale
    return noise1, noise2


#: A negative eigenvalue of the fresh-draw covariance within this many standard
#: errors of the transmitted covariance is sampling noise, clipped to 0.
#: Calibrated on constant C with b_m = 1 (no structural slack, N = 1000), where
#: each clip leaves Var(C) slightly above target and the next generation copies
#: it, so the excess accumulates: the largest ratio was 5.96 over 1000 seeds at
#: G_sim = 8, 7.23 over 300 at G_sim = 20, and 8.71 over 200 at G_sim = 50.
CLIP_SE_MULTIPLIER = 12.0


#: The mates' C cross-covariance term counts as structural (assortative
#: mating), not sampling noise, beyond this many of its standard errors.
CROSS_SE_MULTIPLIER = 4.0


def _transmitted_c(
    rng: np.random.Generator,
    pheno: np.ndarray,
    parents: np.ndarray,
    household_ids: np.ndarray,
    n_hh: int,
    target: np.ndarray,
    c_transmission: CTransmission,
) -> tuple[np.ndarray, np.ndarray, tuple[CClip, ...]]:
    """Return offspring (C1, C2) = transmitted part + a household draw of covariance ``target - S``, and the clips.

    ``S`` is the realized covariance of the transmitted parts over the
    offspring, so the configured ``target`` is the total C covariance in
    expectation. ``S`` splits exactly into the parents' own terms
    ``B_m V_m B_m + B_f V_f B_f`` and the mates' cross term
    ``B_m V_mf B_f + B_f V_fm B_m``. Where ``target - S`` has a negative
    eigenvalue (eigenvector ``v``):

    - a cross term above ``CROSS_SE_MULTIPLIER`` standard errors along ``v``
      (``SE = 2 sqrt(v'B_m V_m B_m v * v'B_f V_f B_f v / n_eff)``) is
      assortative mating pushing ``S`` past the target, and raises;
    - otherwise an excess within ``CLIP_SE_MULTIPLIER`` standard errors of the
      own terms (``SE = sqrt(2 / n_eff) * v'(own)v``) is sampling noise,
      clipped to 0 and returned as a :class:`CClip`; a larger one raises.

    ``n_eff`` is the Kish effective number of households over offspring counts.
    """
    parent_c = pheno[:, [_C1, _C2]]
    mother_c, father_c = parent_c[parents[:, 0]], parent_c[parents[:, 1]]
    B_m, B_f = np.diag(c_transmission.maternal), np.diag(c_transmission.paternal)
    cov = np.cov(np.hstack([mother_c, father_c]), rowvar=False)
    own_m, own_f = B_m @ cov[:2, :2] @ B_m, B_f @ cov[2:, 2:] @ B_f
    half_cross = B_m @ cov[:2, 2:] @ B_f
    cross = half_cross + half_cross.T
    realized = own_m + own_f + cross
    eigvals, eigvecs = np.linalg.eigh(target - realized)
    weights = np.bincount(household_ids, minlength=n_hh) / len(household_ids)
    inv_n_eff = float(np.sum(weights**2))
    clips: list[CClip] = []
    for j in np.flatnonzero(eigvals < 0):
        v, excess = eigvecs[:, j], float(-eigvals[j])
        cross_v = float(v @ cross @ v)
        cross_se = 2.0 * np.sqrt(float(v @ own_m @ v) * float(v @ own_f @ v) * inv_n_eff)
        if cross_v > max(CROSS_SE_MULTIPLIER * cross_se, 1e-12):
            raise ValueError(
                f"C transmission: mates' correlated C (assortative mating) adds {cross_v:.4g} to the "
                f"transmitted C covariance {realized.round(6).tolist()}, past the configured "
                f"{target.round(6).tolist()} by {excess:.4g}; lower the coefficients or the assortment (ADR 0025)"
            )
        tolerance = max(CLIP_SE_MULTIPLIER * np.sqrt(2.0 * inv_n_eff) * float(v @ (own_m + own_f) @ v), 1e-12)
        if excess > tolerance:
            raise ValueError(
                f"C transmission: the transmitted C covariance {realized.round(6).tolist()} exceeds the "
                f"configured {target.round(6).tolist()} (fresh-draw eigenvalue {-excess:.4g}, sampling "
                f"tolerance {tolerance:.4g}); lower the coefficients or raise C (ADR 0025)"
            )
        logger.warning(
            "C transmission: clipped fresh-draw eigenvalue %.4g to 0 (tolerance %.4g); realized Var(C) exceeds the "
            "configured C by that much",
            -excess,
            tolerance,
        )
        clips.append(CClip(excess=excess, tolerance=float(tolerance), direction=tuple(v.round(6).tolist())))
    loadings = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
    fresh = rng.standard_normal((n_hh, 2)) @ loadings.T
    c = mother_c @ B_m + father_c @ B_f + fresh[household_ids]
    return c[:, 0], c[:, 1], tuple(clips)


def reproduce(
    rng: np.random.Generator,
    pheno: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    sd_A1: float,
    sd_E1: float,
    sd_C1: float,
    sd_A2: float,
    sd_E2: float,
    sd_C2: float,
    rA: float,
    rC: float,
    rE: float = 0.0,
    c_transmission: CTransmission | None = None,
    mendelian_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[CClip, ...]]:
    """Simulate offspring phenotypes from parents for two correlated traits.

    Additive genetic values are inherited as midparent + Mendelian noise.
    Common environment (C) is drawn per household (one household per
    mother). Without C transmission that draw is the whole of C, so
    householdmates share C and parents and children do not. With it, C is
    ``b_m * C_mother + b_f * C_father`` plus the household draw, sized so the
    configured C stays the total (see :func:`_transmitted_c`, ADR 0025).
    Unique environment (E) is drawn independently per individual.

    Args:
        rng: numpy random generator
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        household_ids: (n,) array mapping each offspring to a household
        sd_A1: standard deviation of A for trait 1
        sd_E1: standard deviation of E for trait 1
        sd_C1: standard deviation of C for trait 1
        sd_A2: standard deviation of A for trait 2
        sd_E2: standard deviation of E for trait 2
        sd_C2: standard deviation of C for trait 2
        rA: genetic correlation between traits
        rC: common environment correlation between traits
        rE: unique environment correlation between traits
        c_transmission: C transmission coefficients; None or all zero is
            the fresh household draw, identical to the code before ADR 0025.
        mendelian_scale: per-offspring ``sqrt(1 - F_bar)`` applied to the
            Mendelian noise (ADR 0033), or None for the unscaled draw.

    Returns:
        offspring: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex_offspring: (n,) array of sex values (0=female, 1=male)
        clips: under C transmission, one record per fresh-draw eigenvalue
            clipped to 0 (see :func:`_transmitted_c`); otherwise empty

    Raises:
        ValueError: under C transmission, when the realized transmitted
            covariance exceeds the configured C by more than sampling allows.
    """
    n = len(parents)

    # Sex assignment
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # Additive genetic: midparent + correlated Mendelian noise
    mp1 = _midparent(pheno[:, _A1], parents)
    mp2 = _midparent(pheno[:, _A2], parents)

    noise1, noise2 = generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA, mendelian_scale)
    a1_offspring = mp1 + noise1
    a2_offspring = mp2 + noise2

    # Common environment: a draw per household (household_ids are 0-based
    # contiguous from _mating_standard()), plus under C transmission the parents' C.
    n_hh = int(household_ids.max()) + 1
    clips: tuple[CClip, ...] = ()
    if c_transmission is None or not c_transmission.active:
        hh_c1, hh_c2 = generate_correlated_components(rng, n_hh, sd_C1, sd_C2, rC)
        c1_offspring = hh_c1[household_ids]
        c2_offspring = hh_c2[household_ids]
    else:
        c1_offspring, c2_offspring, clips = _transmitted_c(
            rng, pheno, parents, household_ids, n_hh, c_covariance(sd_C1**2, sd_C2**2, rC), c_transmission
        )

    # Unique environment: correlated via rE; independent when rE=0
    if rE != 0:
        e1_offspring, e2_offspring = generate_correlated_components(rng, n, sd_E1, sd_E2, rE)
    else:
        e1_offspring = rng.normal(size=n, loc=0, scale=sd_E1)
        e2_offspring = rng.normal(size=n, loc=0, scale=sd_E2)

    # MZ twins share A values and sex for both traits
    if len(twins) > 0:
        a1_offspring[twins[:, 1]] = a1_offspring[twins[:, 0]]
        a2_offspring[twins[:, 1]] = a2_offspring[twins[:, 0]]
        sex_offspring[twins[:, 1]] = sex_offspring[twins[:, 0]]

    offspring = np.stack(  # column order: _A1 .. _E2
        [
            a1_offspring,
            c1_offspring,
            e1_offspring,
            a2_offspring,
            c2_offspring,
            e2_offspring,
        ],
        axis=-1,
    )

    return offspring, sex_offspring, clips
