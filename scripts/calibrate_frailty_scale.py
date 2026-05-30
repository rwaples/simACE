r"""Monte-Carlo calibrator for the Weibull ``scale`` of plain-frailty scenarios.

Plain ``frailty`` has no ``prevalence`` knob — case fraction is emergent from
the hazard. This script solves for the Weibull ``scale`` that makes the gen-0,
fully-observed cohort's realized prevalence hit a documented target.

It reuses the real ``compute_event_times`` kernel so the onset math is
identical to the simulator, and reproduces the gen-0 ``affected`` identity
from ``simace/censoring/censor.py``:

    affected = (t <= max_age) AND (t <= death_age)

drawing sex and competing death internally so ``beta_sex`` and mortality are
handled by construction. Liability is N(0,1) (gen-0 founders, no assortative
mating), matching the ``standardize: global`` limit where scaled_beta == beta.

Weibull-only by design: every tier-B/C plain-frailty scenario is Weibull; the
lognormal scenarios are handled as cure_frailty (threshold sets K, no solve).

Usage:
    # Solve one trait:
    python scripts/calibrate_frailty_scale.py \\
        --rho 0.8 --beta 1.0 --beta-sex 0.0 \\
        --death-scale 164 --death-rho 2.73 --target-prev 0.10

    # Back-test: predict prevalence at a given (current) scale, no solve:
    python scripts/calibrate_frailty_scale.py \\
        --rho 0.8 --beta 1.0 --death-scale 164 --death-rho 2.73 \\
        --predict-scale 2160

    # Solve every tier-B/C scenario and emit a paste-ready YAML block:
    python scripts/calibrate_frailty_scale.py --all
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import brentq

from simace.phenotype.hazards import compute_event_times

DEFAULT_N = 2_000_000
DEFAULT_SEED = 12345
MAX_AGE = 80.0


def _draw(n: int, seed: int, beta_sex: float, death_scale: float, death_rho: float):
    """Pre-draw the per-sample randomness held fixed across the scale solve.

    Returns ``(liability, neg_log_u_after_sex, death_age)``. Holding these
    fixed makes the prevalence-vs-scale objective deterministic so brentq
    converges on a smooth monotone function.
    """
    rng = np.random.default_rng(seed)
    liability = rng.standard_normal(n)
    sex = rng.binomial(n=1, p=0.5, size=n).astype(np.float64)
    neg_log_u = rng.exponential(size=n)
    if beta_sex != 0.0:
        neg_log_u = neg_log_u / np.exp(beta_sex * sex)
    u_death = 1.0 - rng.uniform(size=n)
    death_age = death_scale * (-np.log(u_death)) ** (1.0 / death_rho)
    return liability, neg_log_u, death_age


def gen0_prevalence(scale: float, rho: float, beta: float, draws, max_age: float = MAX_AGE) -> float:
    """Mean gen-0 ``affected`` fraction for a Weibull scale, given fixed draws."""
    liability, neg_log_u, death_age = draws
    t = compute_event_times(neg_log_u, liability, 0.0, beta, "weibull", {"scale": scale, "rho": rho})
    affected = (t <= max_age) & (t <= death_age)
    return float(affected.mean())


def solve_scale(
    rho: float,
    beta: float,
    beta_sex: float,
    death_scale: float,
    death_rho: float,
    target_prev: float,
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    max_age: float = MAX_AGE,
) -> float:
    """Root-find the Weibull ``scale`` giving ``gen0_prevalence == target_prev``.

    Solves in log10(scale) for conditioning: prevalence is monotone
    decreasing in scale (larger scale -> later onset -> fewer cases by
    ``max_age``), so a sign flip is guaranteed across a wide bracket.
    """
    draws = _draw(n, seed, beta_sex, death_scale, death_rho)

    def objective(log10_scale: float) -> float:
        return gen0_prevalence(10.0**log10_scale, rho, beta, draws, max_age) - target_prev

    log10_scale = brentq(objective, -3.0, 9.0, xtol=1e-4, rtol=1e-6)
    return 10.0**log10_scale


# --- tier-B/C scenario table (plain frailty, MC-calibrated scale) -----------
# Each row: (scenario, trait, rho, beta, beta_sex, death_scale, death_rho,
# target_prev). A/C/E and the *original* scale are irrelevant to the solve
# (marginal L ~ N(0,1); scale is the unknown), so the solved value depends
# only on (rho, beta, beta_sex, death_scale, death_rho, target_prev). Rows
# sharing that tuple MUST get the identical scale -> solve once per tuple.
_TABLE: list[tuple[str, int, float, float, float, float, float, float]] = [
    # Tier C — onset-agnostic (heritability series share one hazard/target)
    ("herit_low", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("herit_low", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.20),
    ("herit_moderate", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("herit_moderate", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.20),
    ("herit_high", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("herit_high", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.20),
    # Prevalence series — finally a real sweep
    ("prev_rare", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.02),
    ("prev_rare", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.02),
    ("prev_moderate", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("prev_moderate", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.15),
    ("prev_common", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.35),
    ("prev_common", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.40),
    # Censoring/mortality contrast — no_mortality uses death_scale 100000
    ("censoring_no_mortality", 1, 0.8, 1.0, 0.0, 100000.0, 2.73, 0.10),
    ("censoring_no_mortality", 2, 1.2, 1.5, 0.0, 100000.0, 2.73, 0.20),
    ("censoring_with_mortality", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("censoring_with_mortality", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.20),
    ("stress_shared_env_common", 1, 0.8, 1.0, 0.0, 164.0, 2.73, 0.35),
    ("stress_shared_env_common", 2, 1.2, 1.5, 0.0, 164.0, 2.73, 0.40),
    # Tier B — late-onset (high rho; survives the solve)
    ("onset_late", 1, 2.5, 1.0, 0.0, 164.0, 2.73, 0.10),
    ("onset_late", 2, 2.2, 1.5, 0.0, 164.0, 2.73, 0.20),
    ("stress_high_herit_late_rare", 1, 2.5, 1.0, 0.0, 164.0, 2.73, 0.02),
    ("stress_high_herit_late_rare", 2, 2.2, 1.5, 0.0, 164.0, 2.73, 0.02),
]


def solve_all(*, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict[str, dict[int, float]]:
    """Solve every table row, deduplicating by the calibration tuple."""
    cache: dict[tuple, float] = {}
    out: dict[str, dict[int, float]] = {}
    for scenario, trait, rho, beta, beta_sex, dscale, drho, tgt in _TABLE:
        key = (rho, beta, beta_sex, dscale, drho, tgt)
        if key not in cache:
            cache[key] = solve_scale(rho, beta, beta_sex, dscale, drho, tgt, n=n, seed=seed)
        out.setdefault(scenario, {})[trait] = cache[key]
    return out


def _emit_yaml(solved: dict[str, dict[int, float]]) -> None:
    """Print a paste-ready summary of solved scales per scenario/trait."""
    print("# Solved Weibull scales (paste each into the matching trait's params.scale):")
    for scenario, traits in solved.items():
        print(f"{scenario}:")
        for trait in sorted(traits):
            print(f"  trait{trait}: scale: {traits[trait]:.4g}")


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="Solve every tier-B/C scenario and emit YAML.")
    p.add_argument("--rho", type=float, help="Weibull shape.")
    p.add_argument("--beta", type=float, default=1.0, help="Liability coefficient on log-hazard.")
    p.add_argument("--beta-sex", type=float, default=0.0, help="Sex coefficient on log-hazard.")
    p.add_argument("--death-scale", type=float, default=164.0, help="Competing-death Weibull scale.")
    p.add_argument("--death-rho", type=float, default=2.73, help="Competing-death Weibull shape.")
    p.add_argument("--max-age", type=float, default=MAX_AGE, help="gen-0 upper observation bound.")
    p.add_argument("--target-prev", type=float, help="Target gen-0 prevalence to solve for.")
    p.add_argument(
        "--predict-scale",
        type=float,
        default=None,
        help="Back-test mode: print gen-0 prevalence at this scale instead of solving.",
    )
    p.add_argument("--n", type=int, default=DEFAULT_N, help="Monte-Carlo sample size.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed.")
    args = p.parse_args()

    if args.all:
        _emit_yaml(solve_all(n=args.n, seed=args.seed))
        return 0

    if args.rho is None:
        p.error("--rho is required (or use --all)")

    if args.predict_scale is not None:
        draws = _draw(args.n, args.seed, args.beta_sex, args.death_scale, args.death_rho)
        prev = gen0_prevalence(args.predict_scale, args.rho, args.beta, draws, args.max_age)
        print(f"predicted gen-0 prevalence at scale={args.predict_scale:g}: {prev:.4f}")
        return 0

    if args.target_prev is None:
        p.error("--target-prev is required when solving (or use --predict-scale / --all)")

    scale = solve_scale(
        args.rho,
        args.beta,
        args.beta_sex,
        args.death_scale,
        args.death_rho,
        args.target_prev,
        n=args.n,
        seed=args.seed,
        max_age=args.max_age,
    )
    print(f"scale = {scale:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
