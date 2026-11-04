"""Measure how far frailty and adult.ltm event times move the wrong way when liability or K rises.

Backs the ``_MIN_STEP`` floor in ``tests/phenotype/models/test_monotonicity_properties.py``.
Each frailty distribution gets 1.8M liability pairs (3 betas x 3 step kinds x
200_000: an exponential shift, one ulp, and +1e-9); adult.ltm gets 4.8M
liability pairs (4 K x 3 betas x 2 step kinds) and 6.6M prevalence pairs.
Seed 0. Prints the violation count and worst relative increase per family.

    pixi run python tools/probe_event_time_monotonicity.py
"""

import numpy as np

from simace.phenotype.models.adult import AdultModel
from simace.phenotype.models.frailty import FrailtyModel

PARAMS = {
    "weibull": {"scale": 2160.0, "rho": 0.8},
    "exponential": {"rate": 0.01},
    "gompertz": {"rate": 1e-4, "gamma": 0.08},
    "lognormal": {"mu": 4.0, "sigma": 1.0},
    "loglogistic": {"scale": 60.0, "shape": 4.0},
    "gamma": {"shape": 2.0, "scale": 30.0},
}
rng = np.random.default_rng(0)
n = 200_000
gen = np.repeat(np.arange(4), n // 4)
sex = rng.integers(0, 2, n)
for dist, p in PARAMS.items():
    viol = 0
    worst = 0.0
    for beta in (0.3, 1.0, 3.0):
        m = FrailtyModel(distribution=dist, hazard_params=p, beta=beta, beta_sex=0.4)
        L = rng.normal(0, 1.5, n)
        for Lp in (L + rng.exponential(0.5, n), np.nextafter(L, np.inf), L + 1e-9):
            t = m.simulate(L, seed=7, standardize="none", sex=sex, generation=gen)
            tp = m.simulate(Lp, seed=7, standardize="none", sex=sex, generation=gen)
            bad = tp > t
            viol += int(bad.sum())
            if bad.any():
                worst = max(worst, float(np.max((tp[bad] - t[bad]) / t[bad])))
    print(f"frailty {dist:12s} n_pairs={3 * 3 * n} violations={viol} worst_rel={worst:.3g}")

viol_L = viol_K = 0
worst_L = 0.0
nK = 0
for K in (0.01, 0.1, 0.5, 0.9):
    for beta in (0.3, 1.0, 3.0):
        m = AdultModel(method="ltm", prevalence=K, beta=beta, beta_sex=0.4)
        L = rng.normal(0, 1.5, n)
        for Lp in (L + rng.exponential(0.5, n), np.nextafter(L, np.inf)):
            t = m.simulate(L, seed=7, standardize="none", sex=sex, generation=gen)
            tp = m.simulate(Lp, seed=7, standardize="none", sex=sex, generation=gen)
            bad = tp > t
            viol_L += int(bad.sum())
            if bad.any():
                worst_L = max(worst_L, float(np.max((tp[bad] - t[bad]) / t[bad])))
        for Kp in (K * 1.5, np.nextafter(K, 1.0), K + 1e-10):
            if Kp >= 1:
                continue
            mp = AdultModel(method="ltm", prevalence=Kp, beta=beta, beta_sex=0.4)
            t = m.simulate(L, seed=7, standardize="none", sex=sex, generation=gen)
            tp = mp.simulate(L, seed=7, standardize="none", sex=sex, generation=gen)
            viol_K += int((tp > t).sum())
            nK += n
print(f"adult.ltm L-monotone violations={viol_L} over {4 * 3 * 2 * n} pairs worst_rel={worst_L:.3g}")
print(f"adult.ltm K-monotone violations={viol_K} over {nK} pairs")
