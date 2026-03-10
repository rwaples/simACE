
  # ── EXAMPLES ─────────────────────────────────────────────────────────────

  # ==============================================================================
  # 1) WEIBULL
  # ==============================================================================
  # Hazard: h(t) = (rho / scale) * (t / scale)^(rho - 1)
  #
  # Mean:
  #     mean = scale * Γ(1 + 1/rho)
  #
  # Variance:
  #     var  = scale^2 * [ Γ(1 + 2/rho) - Γ(1 + 1/rho)^2 ]
  #
  # INVERSE (approximate):
  #   Given a desired mean m and shape rho:
  #       scale = m / Γ(1 + 1/rho)
  #
  # NOTE:
  #   - rho > 1  → increasing hazard with age
  #   - rho = 1  → exponential model
  #   - rho < 1  → decreasing hazard
  #
  # Example:
  # hazard_model1: weibull
  # hazard_params1:
  #   scale: 2160
  #   rho: 0.8



  # ==============================================================================
  # 2) EXPONENTIAL
  # ==============================================================================
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
  # hazard_model1: exponential
  # hazard_params1:
  #   rate: 0.02    # mean onset ≈ 50



  # ==============================================================================
  # 3) GOMPERTZ
  # ==============================================================================
  # Hazard: h(t) = rate * exp(gamma * t)
  #
  # Mean:
  #     mean = (1/gamma) * E1(rate/gamma) * exp(rate/gamma)
  #     (E1 = exponential integral)
  #
  # There is no simple closed-form inverse for (rate, gamma).
  #
  # PRACTICAL GUIDELINES (inverse use):
  #   - gamma controls the speed of aging (0.03–0.10 realistic)
  #   - rate sets the initial hazard
  #   - To shift the age of onset higher → decrease rate
  #   - To make onset earlier → increase gamma
  #
  # Example:
  # hazard_model1: gompertz
  # hazard_params1:
  #   rate: 0.00005
  #   gamma: 0.07



  # ==============================================================================
  # 4) LOGNORMAL
  # ==============================================================================
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
  # Example (mean ≈ 65, sd ≈ 5):
  # hazard_model1: lognormal
  # hazard_params1:
  #   mu: 4.160
  #   sigma: 0.076



  # ==============================================================================
  # 5) LOGLOGISTIC
  # ==============================================================================
  # PDF(t) = (shape/scale) * (t/scale)^(shape-1) / (1 + (t/scale)^shape)^2
  # Hazard increases → peaks → decreases.
  #
  # Median:
  #     median = scale * ( 2^(1/shape) - 1 )^(1/shape)
  #
  # Mean (exists only if shape > 1):
  #     mean = scale * π/shape / sin(π/shape)
  #
  # INVERSE (for shape > 1):
  #   Given desired mean m:
  #       scale = m * (shape / π) * sin(π/shape)
  #
  # NOTE:
  #   shape > 1 → hazard increases then decreases
  #   shape = 1 → logistic distribution (monotone)
  #
  # Example:
  # hazard_model1: loglogistic
  # hazard_params1:
  #   scale: 50
  #   shape: 3.5



  # ==============================================================================
  # 6) GAMMA (Accelerated‑Time Model)
  # ==============================================================================
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
  # hazard_model1: gamma
  # hazard_params1:
  #   shape: 36.0       # (60/10)^2
  #   scale: 1.6667     # (10^2)/60
  ################################################################################

