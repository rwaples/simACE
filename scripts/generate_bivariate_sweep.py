#!/usr/bin/env python
"""Generate factorial sweep scenarios for bivariate PA-FGRS characterization.

Produces YAML scenario entries for all combinations of:
  rA × h² × prevalence × N

Output is appended to config/config.yaml under the bivariate_sweep folder.
"""

from __future__ import annotations

import yaml

# Factorial grid
RA_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7]
H2_LEVELS = [0.2, 0.5, 0.8]
PREV_LEVELS = [0.05, 0.10, 0.20]
N_LEVELS = [1_000, 10_000, 100_000]

# Fixed parameters
DEFAULTS = {
    "folder": "bivariate_sweep",
    "replicates": 3,
    "phenotype_model1": "weibull",
    "phenotype_model2": "weibull",
    "phenotype_params1": {"scale": 2160, "rho": 0.8},
    "phenotype_params2": {"scale": 333, "rho": 1.2},
    "beta1": 1.0,
    "beta2": 1.5,
    "rC": 0.5,
    "G_ped": 6,
    "G_pheno": 3,
    "G_sim": 8,
}


def scenario_name(rA: float, h2: float, prev: float, N: int) -> str:
    """Generate scenario name like biv_rA03_h2050_p010_N10K."""
    rA_str = f"{int(rA * 10):d}"
    h2_str = f"{int(h2 * 100):03d}"
    prev_str = f"{int(prev * 100):03d}"
    if N >= 1_000_000:
        n_str = f"{N // 1_000_000}M"
    elif N >= 1_000:
        n_str = f"{N // 1_000}K"
    else:
        n_str = str(N)
    return f"biv_rA{rA_str}_h2{h2_str}_p{prev_str}_N{n_str}"


def generate_scenarios() -> dict:
    """Generate all factorial scenarios."""
    scenarios = {}
    seed_base = 100_000

    for i_rA, rA in enumerate(RA_LEVELS):
        for i_h2, h2 in enumerate(H2_LEVELS):
            for i_prev, prev in enumerate(PREV_LEVELS):
                for i_N, N in enumerate(N_LEVELS):
                    name = scenario_name(rA, h2, prev, N)
                    seed = seed_base + i_rA * 1000 + i_h2 * 100 + i_prev * 10 + i_N

                    C = round((1.0 - h2) * 0.4, 4)  # C = 40% of non-genetic

                    scenario = {
                        **DEFAULTS,
                        "seed": seed,
                        "N": N,
                        "rA": rA,
                        "A1": h2,
                        "C1": C,
                        "A2": h2,
                        "C2": C,
                        "prevalence1": prev,
                        "prevalence2": prev,
                    }
                    scenarios[name] = scenario

    return scenarios


def main():
    scenarios = generate_scenarios()
    print(f"# Generated {len(scenarios)} bivariate sweep scenarios")
    print(f"# Grid: rA={RA_LEVELS} x h2={H2_LEVELS} x prev={PREV_LEVELS} x N={N_LEVELS}")
    print()

    # Output as YAML fragment
    output = yaml.dump(
        {"scenarios": scenarios},
        default_flow_style=False,
        sort_keys=False,
    )
    print(output)

    print(f"\n# Total: {len(scenarios)} scenarios")
    print(f"# With {DEFAULTS['replicates']} replicates each: {len(scenarios) * DEFAULTS['replicates']} runs")


if __name__ == "__main__":
    main()
