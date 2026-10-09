"""Baseline hazard computation for parametric survival models."""

import numpy as np

__all__ = ["compute_hazard_terms"]
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgauss, norm

from simace.core.phenotype_keys import exponential_rate


def compute_hazard_terms(
    model: str,
    t: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-baseline-hazard and cumulative baseline hazard.

    Returns (const, H_base) where:
        const  = log h0(t)   — event term:    delta * (const + beta*L)
        H_base = H0(t)       — survival term: H_base * exp(beta*L)

    Supported models and their required params:
        "weibull"     : {"scale": s, "rho": rho}
        "exponential" : {"rate": lam}  or  {"scale": s}
        "gompertz"    : {"rate": b, "gamma": g}
        "lognormal"   : {"mu": mu, "sigma": sigma}
        "loglogistic" : {"scale": alpha, "shape": k}
        "gamma"       : {"shape": k, "scale": theta}
        "first_passage": {"drift": mu, "shape": lam}

    Raises:
        ValueError: unknown model name or missing required parameter.
    """
    t = np.asarray(t, dtype=np.float64)

    if model == "weibull":
        s, rho = params["scale"], params["rho"]
        log_t = np.log(t)
        const = np.log(rho) - rho * np.log(s) + (rho - 1) * log_t
        # an exponent beyond ~709 overflows to inf, which is the correct limit (S0 -> 0)
        with np.errstate(over="ignore"):
            H_base = np.exp(rho * (log_t - np.log(s)))

    elif model == "exponential":
        lam = exponential_rate(params)
        const = np.full_like(t, np.log(lam))
        H_base = lam * t

    elif model == "gompertz":
        # h0(t)=b·exp(g·t),  H0(t)=(b/g)·(exp(g·t)−1)  [expm1 stable near 0]
        b, g = params["rate"], params["gamma"]
        const = np.log(b) + g * t
        if abs(g) < 1e-12:
            H_base = b * t
        else:
            # g*t beyond ~709 overflows to inf, which is the correct limit (S0 -> 0)
            with np.errstate(over="ignore"):
                H_base = (b / g) * np.expm1(g * t)

    elif model == "lognormal":
        # H0(t)=−log S0(t)=−norm.logsf(z),  z=(log t−mu)/sigma
        mu, sigma = params["mu"], params["sigma"]
        log_t = np.log(t)
        z = (log_t - mu) / sigma
        log_S0 = norm.logsf(z)  # stable complementary log-CDF
        log_f0 = -0.5 * z**2 - 0.5 * np.log(2 * np.pi) - np.log(sigma) - log_t
        const = log_f0 - log_S0
        H_base = -log_S0

    elif model == "loglogistic":
        # H0(t)=log(1+(t/α)^k)=log(exp(0)+exp(u)), u=k·log(t/α)
        # logaddexp is the stable log-sum-exp: no overflow for large u
        alpha, k = params["scale"], params["shape"]
        u = k * (np.log(t) - np.log(alpha))
        H_base = np.logaddexp(0.0, u)
        const = np.log(k) - np.log(alpha) + (k - 1) * (np.log(t) - np.log(alpha)) - H_base

    elif model == "gamma":
        # H0(t)=−gamma_dist.logsf(t; shape=k, scale=θ)  [stable log-survival]
        k, theta = params["shape"], params["scale"]
        log_f0 = (k - 1) * np.log(t) - t / theta - k * np.log(theta) - gammaln(k)
        log_S0 = gamma_dist.logsf(t, a=k, scale=theta)
        const = log_f0 - log_S0
        H_base = -log_S0

    elif model == "first_passage":
        # First-passage time of Wiener process Y(t)=y0+mu*t+W(t) hitting 0.
        # FPT ~ Inverse Gaussian.  drift<0: everyone hits; drift>0: cure fraction.
        drift_val, lam = params["drift"], params["shape"]
        if drift_val == 0.0:
            raise ValueError("first_passage drift must be non-zero")
        y0 = np.sqrt(lam)
        ig_mean = y0 / abs(drift_val)
        # scipy parameterization: invgauss(mu=ig_mean/lam, scale=lam)
        rv = invgauss(mu=ig_mean / lam, scale=lam)
        if drift_val < 0:
            # Everyone hits — standard IG distribution
            log_sf = rv.logsf(t)
            const = rv.logpdf(t) - log_sf
            H_base = -log_sf
        else:
            # Defective distribution — cure fraction exists
            p_hit = np.exp(-2.0 * y0 * drift_val)
            log_S_def = np.log1p(-p_hit * rv.cdf(t))
            const = np.log(p_hit) + rv.logpdf(t) - log_S_def
            H_base = -log_S_def

    else:
        raise ValueError(
            f"Unknown hazard model '{model}'. "
            "Supported: 'weibull','exponential','gompertz','lognormal',"
            "'loglogistic','gamma','first_passage'."
        )
    return const, H_base
