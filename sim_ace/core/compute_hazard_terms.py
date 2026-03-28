import numpy as np
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


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

    Raises:
        ValueError: unknown model name or missing required parameter.
    """
    t = np.asarray(t, dtype=np.float64)

    if model == "weibull":
        s, rho = params["scale"], params["rho"]
        log_t = np.log(t)
        const = np.log(rho) - rho * np.log(s) + (rho - 1) * log_t
        H_base = np.exp(rho * (log_t - np.log(s)))

    elif model == "exponential":
        if "rate" in params:
            lam = params["rate"]
        elif "scale" in params:
            lam = 1.0 / params["scale"]
        else:
            raise ValueError("exponential: need 'rate' or 'scale'")
        const = np.full_like(t, np.log(lam))
        H_base = lam * t

    elif model == "gompertz":
        # h0(t)=b·exp(g·t),  H0(t)=(b/g)·(exp(g·t)−1)  [expm1 stable near 0]
        b, g = params["rate"], params["gamma"]
        const = np.log(b) + g * t
        H_base = b * t if abs(g) < 1e-12 else (b / g) * np.expm1(g * t)

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
        # H0(t)=log(1+(t/α)^k)=log1p(exp(u)), u=k·log(t/α)
        # LSE trick: for large u, log1p(exp(u))≈u
        alpha, k = params["scale"], params["shape"]
        u = k * (np.log(t) - np.log(alpha))
        H_base = np.where(u > 30.0, u, np.log1p(np.exp(u)))
        const = np.log(k) - np.log(alpha) + (k - 1) * (np.log(t) - np.log(alpha)) - H_base

    elif model == "gamma":
        # H0(t)=−gamma_dist.logsf(t; shape=k, scale=θ)  [stable log-survival]
        k, theta = params["shape"], params["scale"]
        log_f0 = (k - 1) * np.log(t) - t / theta - k * np.log(theta) - gammaln(k)
        log_S0 = gamma_dist.logsf(t, a=k, scale=theta)
        const = log_f0 - log_S0
        H_base = -log_S0

    else:
        raise ValueError(
            f"Unknown hazard model '{model}'. "
            "Supported: 'weibull','exponential','gompertz','lognormal','loglogistic','gamma'."
        )
    return const, H_base
