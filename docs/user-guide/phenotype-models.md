
# ACE Phenotype Models

Four model families convert pedigree liabilities (L = A + C + E) to
age-of-onset times. Set independently per trait via `phenotype_model1`/`phenotype_model2`
in config (default: `frailty`).

```yaml
phenotype_model1: frailty         # frailty / cure_frailty / adult / first_passage
phenotype_params1:
  distribution: weibull           # for frailty/cure_frailty: weibull/exponential/gompertz/lognormal/loglogistic/gamma
                                  # for adult: method: ltm/cox
```

All models produce raw event times `t1`, `t2` which are then censored by
ACE's downstream censoring pipeline (age-window + Weibull competing-risk mortality).

## Index

| # | Model | Sub-key |
|---|-------|---------|
| 1 | `frailty` (default) | `distribution:` |
|   | 1a Weibull | `distribution: weibull` |
|   | 1b Exponential | `distribution: exponential` |
|   | 1c Gompertz | `distribution: gompertz` |
|   | 1d Lognormal | `distribution: lognormal` |
|   | 1e Loglogistic | `distribution: loglogistic` |
|   | 1f Gamma | `distribution: gamma` |
| 2 | `cure_frailty` | `distribution:` |
| 3 | `adult` (LTM) | `method: ltm` |
| 4 | `adult` (Cox) | `method: cox` |
| 5 | `first_passage` | (none) |


# ==============================================================================
# 1) FRAILTY MODEL (default)
# ==============================================================================
# Proportional hazards frailty model with pluggable baseline hazard.
#
# Model (per trait):
#     L        = A + C + E               (liability from pedigree)
#     z        = exp(beta * L_std)       (frailty / hazard multiplier)
#     S(t | z) = exp(-H0(t) * z)        (conditional survival)
#     t        = H0^{-1}(-log(U) / z)   where U ~ Uniform(0, 1]
#
# L_std = (L - mean(L)) / std(L) when standardize=True.
# beta controls the strength of liability on hazard (higher = stronger).
# An optional sex covariate multiplies the hazard by exp(beta_sex * sex).
#
# Config parameters:
#     beta1, beta2:               liability effect on log-hazard
#     beta_sex1, beta_sex2:       sex covariate effect (0 = no effect)
#     phenotype_model1, phenotype_model2: model family name
#     phenotype_params1, phenotype_params2: distribution + model-specific parameters
#
# Example:
# phenotype_model1: frailty
# beta1: 1.0
# phenotype_params1:
#   distribution: weibull
#   scale: 2160
#   rho: 0.8


# ------------------------------------------------------------------------------
# Baseline hazard distributions (frailty model only)
# ------------------------------------------------------------------------------

  # ============================================================================
  # 1a) WEIBULL
  # ============================================================================
  # Hazard: h(t) = (rho / scale) * (t / scale)^(rho - 1)
  #
  # Mean:
  #     mean = scale * Gamma(1 + 1/rho)
  #
  # Variance:
  #     var  = scale^2 * [ Gamma(1 + 2/rho) - Gamma(1 + 1/rho)^2 ]
  #
  # INVERSE (approximate):
  #   Given a desired mean m and shape rho:
  #       scale = m / Gamma(1 + 1/rho)
  #
  # NOTE:
  #   - rho > 1  -> increasing hazard with age
  #   - rho = 1  -> exponential model
  #   - rho < 1  -> decreasing hazard
  #
  # Example:
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: weibull
  #   scale: 2160
  #   rho: 0.8



  # ============================================================================
  # 1b) EXPONENTIAL
  # ============================================================================
  # Hazard: h(t) = rate (constant)
  #
  # Mean:
  #     mean = 1 / rate
  #
  # Variance:
  #     var = 1 / rate^2
  #
  # INVERSE:
  #   rate = 1 / mean
  #
  # Example:
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: exponential
  #   rate: 0.02    # mean onset ~ 50



  # ============================================================================
  # 1c) GOMPERTZ
  # ============================================================================
  # Hazard: h(t) = rate * exp(gamma * t)
  #
  # Mean:
  #     mean = (1/gamma) * E1(rate/gamma) * exp(rate/gamma)
  #     (E1 = exponential integral)
  #
  # There is no simple closed-form inverse for (rate, gamma).
  #
  # PRACTICAL GUIDELINES (inverse use):
  #   - gamma controls the speed of aging (0.03-0.10 realistic)
  #   - rate sets the initial hazard
  #   - To shift the age of onset higher -> decrease rate
  #   - To make onset earlier -> increase gamma
  #
  # Example:
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: gompertz
  #   rate: 0.00005
  #   gamma: 0.07



  # ============================================================================
  # 1d) LOGNORMAL
  # ============================================================================
  # Model: log(age_at_event) ~ Normal(mu, sigma)
  #
  # Direct:
  #     mean  = exp(mu + sigma^2 / 2)
  #     var   = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
  #
  # INVERSE (to get mu, sigma from desired age mean m and std s):
  #     sigma = sqrt( ln( 1 + (s/m)^2 ) )
  #     mu    = ln(m) - 0.5 * sigma^2
  #
  #
  # Example (mean ~ 65, sd ~ 5):
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: lognormal
  #   mu: 4.160
  #   sigma: 0.076



  # ============================================================================
  # 1e) LOGLOGISTIC
  # ============================================================================
  # PDF(t) = (shape/scale) * (t/scale)^(shape-1) / (1 + (t/scale)^shape)^2
  # Hazard increases -> peaks -> decreases.
  #
  # Median:
  #     median = scale
  #
  # Mean (exists only if shape > 1):
  #     mean = scale * pi/shape / sin(pi/shape)
  #
  # INVERSE (for shape > 1):
  #   Given desired mean m:
  #       scale = m * (shape / pi) * sin(pi/shape)
  #
  # NOTE:
  #   shape > 1 -> hazard increases then decreases
  #   shape = 1 -> logistic distribution (monotone)
  #
  # Example:
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: loglogistic
  #   scale: 50
  #   shape: 3.5



  # ============================================================================
  # 1f) GAMMA (Accelerated-Time Model)
  # ============================================================================
  # age_at_event ~ Gamma(shape, scale)
  #
  # Direct:
  #     mean = shape * scale
  #     var  = shape * scale^2
  #
  # INVERSE (exact):
  #     shape = (mean / sd)^2
  #     scale = (sd^2) / mean
  #
  # Example (mean ~ 60, sd ~ 10):
  # phenotype_model1: frailty
  # phenotype_params1:
  #   distribution: gamma
  #   shape: 36.0       # (60/10)^2
  #   scale: 1.6667     # (10^2)/60



# ==============================================================================
# 2) MIXTURE CURE FRAILTY MODEL (cure_frailty)
# ==============================================================================
# Mixture cure model: liability threshold determines WHO gets the disorder,
# then a proportional hazards frailty model determines WHEN (age-of-onset)
# among cases. Controls are censored at 1e6.
#
# Model (per trait):
#     L_std    = standardize(L)                      (N(0,1) liability)
#     is_case  = L_std > Phi^{-1}(1 - K)             (top K fraction)
#     z        = exp(beta * L_std)                    (frailty, cases only)
#     t_case   = H0^{-1}(-log(U) / z)                (age-at-onset)
#     t_ctrl   = 1e6                                  (censored downstream)
#
# This separates case status (deterministic from liability rank) from
# age-of-onset (stochastic from frailty + baseline hazard), unlike the
# pure frailty model where both are entangled.
#
# The `distribution` key in phenotype_params selects which frailty baseline
# hazard to use (any of the 6 distributions: weibull, exponential,
# gompertz, lognormal, loglogistic, gamma).
#
# Config parameters:
#     prevalence1, prevalence2:        population prevalence K per trait
#     beta1, beta2:                    liability effect on log-hazard (cases only)
#     beta_sex1, beta_sex2:            sex covariate effect (0 = no effect)
#     phenotype_params1/2:
#       distribution: <distribution>   baseline hazard distribution name
#       <distribution params>:         parameters for the chosen distribution
#
# Example:
# phenotype_model1: cure_frailty
# prevalence1: 0.10
# beta1: 1.0
# phenotype_params1:
#   distribution: weibull
#   scale: 2160
#   rho: 0.8



# ==============================================================================
# 3) ADuLT LIABILITY THRESHOLD MODEL (adult, method: ltm)
# ==============================================================================
# From Pedersen et al., Nat Commun 2023.
# Deterministic mapping from liability to age-of-onset via the logistic
# cumulative incidence proportion (CIP) function.
#
# Logistic CIP (Eq. 3):
#     CIP(age) = K / (1 + exp(-k * (age - x0)))
#
# where K = prevalence, x0 = midpoint age, k = growth rate.
#
# Steps:
#   1. Standardize L to N(0,1)
#   2. Case if L > Phi^{-1}(1 - K)    (top K fraction of liability)
#   3. Case age = x0 + (1/k) * log(Phi(-L) / (K - Phi(-L)))
#      i.e., CIP inverse applied to the individual's cumulative risk Phi(-L)
#   4. Controls: t = 1e6 (censored downstream)
#
# Properties:
#   - Deterministic: same liability always maps to same age (no randomness)
#   - Higher liability -> younger onset age among cases
#   - Case rate = prevalence (by construction)
#   - Case onset ages are centered around x0
#
# Parameters:
#   k controls the spread of onset ages:
#     - Small k (e.g. 0.1): wide age-of-onset range (~10-90 years)
#     - Large k (e.g. 0.5): concentrated near x0 (~40-60 years)
#     - The scale parameter of the logistic distribution is 1/k
#
# Config parameters:
#     prevalence1, prevalence2:  population prevalence K per trait
#     phenotype_params1/2:
#       cip_x0:  logistic CIP midpoint age (default 50)
#       cip_k:   logistic CIP growth rate (default 0.2)
#
# Example:
# phenotype_model1: adult
# phenotype_model2: adult
# prevalence1: 0.10
# prevalence2: 0.20
# phenotype_params1:
#   method: ltm
#   cip_x0: 50
#   cip_k: 0.2
# phenotype_params2:
#   method: ltm
#   cip_x0: 50
#   cip_k: 0.2



# ==============================================================================
# 4) ADuLT PROPORTIONAL HAZARDS MODEL (adult, method: cox)
# ==============================================================================
# From Pedersen et al., Nat Commun 2023.
# Weibull(shape=2) proportional hazards with rank-based CIP-to-age mapping.
#
# Steps:
#   1. Standardize L to N(0,1)
#   2. Raw event time: t_raw = sqrt(-log(U) / exp(L)),  U ~ Uniform(0,1)
#      This is a Weibull(shape=2) frailty model with frailty = exp(L).
#   3. Sort all individuals by t_raw (ascending)
#   4. Running CIP: cip_i = rank_i / (N + 1)
#   5. Case if cip_i < K (prevalence)
#   6. Case age = x0 + (1/k) * log(cip_i / (K - cip_i))
#   7. Controls: t = 1e6 (censored downstream)
#
# Properties:
#   - Stochastic: the Weibull noise (U) introduces randomness
#   - Liability affects case status probabilistically via the hazard
#   - Case rate = prevalence (by rank cutoff)
#   - Case onset ages centered around x0
#
# Key difference from adult (ltm):
#   - adult (ltm) is deterministic (liability alone determines case/age)
#   - adult (cox) adds Weibull noise, so two individuals with the same liability
#     may differ in case status and onset age
#
# Config parameters:
#     prevalence1, prevalence2:  population prevalence K per trait
#     phenotype_params1/2:
#       cip_x0:  logistic CIP midpoint age (default 50)
#       cip_k:   logistic CIP growth rate (default 0.2)
#
# Example:
# phenotype_model1: adult
# phenotype_model2: adult
# prevalence1: 0.10
# prevalence2: 0.20
# phenotype_params1:
#   method: cox
#   cip_x0: 50
#   cip_k: 0.2
# phenotype_params2:
#   method: cox
#   cip_x0: 50
#   cip_k: 0.2



# ==============================================================================
# 5) FIRST-PASSAGE TIME MODEL (first_passage)
# ==============================================================================
# Based on Lee & Whitmore (2006) and Aalen & Gjessing (2001).
# A latent Wiener process Y(t) = y0 + drift*t + W(t) represents cumulative
# health degradation. Disease onset occurs at the first time Y(t) <= 0.
# Event times follow an inverse Gaussian distribution.
#
# Unlike frailty models where the unique environment E is a single draw at
# birth, here E operates as an ongoing stochastic process W(t) — Brownian
# motion that accumulates over the life course.
#
# Model (per trait):
#     L_std     = standardize(L)                             (N(0,1) liability)
#     y0_i      = sqrt(shape) * exp(-beta*L_std - beta_sex*sex)  (initial distance)
#     Y_i(t)    = y0_i + drift*t + W_i(t)                   (Wiener process)
#     T_i       = inf{t : Y_i(t) <= 0}                      (first passage)
#
# Liability scales y0 (initial distance to boundary):
#   - Higher liability -> smaller y0 -> closer to boundary -> earlier onset
#   - beta > 0 means higher liability is worse (same convention as frailty)
#   - beta = 0 means liability has no effect on timing
#
# Two drift regimes:
#
#   drift < 0 (toward boundary):
#     Everyone eventually hits. Event times ~ IG(y0/|drift|, y0^2).
#     Mean event time = sqrt(shape) / |drift|.
#
#   drift > 0 (away from boundary):
#     Emergent cure fraction: P(never hit) = 1 - exp(-2*y0*drift).
#     Higher liability -> smaller y0 -> HIGHER probability of hitting
#     AND earlier onset among those who hit. Both susceptibility and
#     timing emerge from a single beta parameter.
#
#   drift = 0: degenerate case (Lévy-distributed FPT with infinite mean).
#
# Inverse Gaussian distribution:
#   T ~ IG(mu, lambda) where:
#     mu     = y0 / |drift|       (mean event time)
#     lambda = y0^2               (shape parameter)
#
#   PDF:  f(t) = sqrt(lambda / (2*pi*t^3)) * exp(-lambda*(t-mu)^2 / (2*mu^2*t))
#   Mean: E[T] = mu
#   Var:  V[T] = mu^3 / lambda
#
# Hazard shape:
#   The IG hazard has an "upside-down bathtub" shape:
#     - Starts at 0, rises to a peak, then decreases
#     - Asymptotes to lambda/(2*mu^2) > 0 (residual risk never vanishes)
#   This is distinct from Weibull (monotone only), Gompertz (monotone
#   increasing), and lognormal (UBT but h->0).
#
# INVERSE (to get drift and shape from desired mean m and CV):
#     shape = m^2 / (m * CV^2)       [since Var = mu^3/lambda, CV^2 = mu/lambda]
#     drift = -sqrt(shape) / m
#
# Config parameters:
#     beta1, beta2:               liability effect on log(y0)
#     beta_sex1, beta_sex2:       sex covariate effect on log(y0) (0 = no effect)
#     phenotype_params1/2:
#       drift:  Wiener drift rate (negative = toward boundary, positive = cure)
#       shape:  y0^2 (initial distance squared; controls spread)
#
# Example (everyone hits, mean onset ~ 1000 time units):
# phenotype_model1: first_passage
# beta1: 1.0
# phenotype_params1:
#   drift: -0.01
#   shape: 100        # y0 = 10, mean = 10/0.01 = 1000
#
# Example (emergent cure fraction):
# phenotype_model1: first_passage
# beta1: 1.0
# phenotype_params1:
#   drift: 0.05
#   shape: 100        # cure fraction ~ 1 - exp(-2*10*0.05) = 63.2%
