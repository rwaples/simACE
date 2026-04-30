# ACE Phenotype Models

Five model families convert pedigree liabilities ($L = A + C + E$) to
age-of-onset times. Set independently per trait via `phenotype_model1` /
`phenotype_model2` in config (default: `frailty`).

```yaml
phenotype_model1: frailty         # frailty / cure_frailty / adult / first_passage
phenotype_params1:
  distribution: weibull           # for frailty/cure_frailty: weibull/exponential/gompertz/lognormal/loglogistic/gamma
                                  # for adult: method: ltm/cox
```

All models produce raw event times `t1`, `t2` which are then censored by
the downstream censoring pipeline (age-window + Weibull competing-risk
mortality).

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

---

## 1. Frailty model (default)

Proportional-hazards frailty model with pluggable baseline hazard.

Per trait:

$$
L = A + C + E \quad\text{(liability from pedigree)}
$$

$$
z = \exp(\beta \cdot L_{\text{std}}) \quad\text{(frailty / hazard multiplier)}
$$

$$
S(t \mid z) = \exp(-H_0(t) \cdot z) \quad\text{(conditional survival)}
$$

$$
t = H_0^{-1}\!\left(\frac{-\log U}{z}\right), \quad U \sim \text{Uniform}(0, 1]
$$

$L_{\text{std}} = (L - \overline{L}) / \mathrm{sd}(L)$ when `standardize: true`.
$\beta$ controls the strength of liability on hazard (higher = stronger).
An optional sex covariate multiplies the hazard by $\exp(\beta_{\text{sex}} \cdot \text{sex})$.

**Config parameters**

| Key | Description |
|---|---|
| `beta1`, `beta2` | Liability effect on log-hazard |
| `beta_sex1`, `beta_sex2` | Sex covariate effect (0 = no effect) |
| `phenotype_model1`, `phenotype_model2` | Model family name |
| `phenotype_params1`, `phenotype_params2` | Distribution + model-specific parameters |

**Example**

```yaml
phenotype_model1: frailty
beta1: 1.0
phenotype_params1:
  distribution: weibull
  scale: 2160
  rho: 0.8
```

### Baseline hazard distributions (frailty model)

#### 1a. Weibull

$$
h(t) = \frac{\rho}{\text{scale}} \left(\frac{t}{\text{scale}}\right)^{\rho - 1}
$$

- Mean: $\text{scale} \cdot \Gamma(1 + 1/\rho)$
- Variance: $\text{scale}^2 \cdot \left[\Gamma(1 + 2/\rho) - \Gamma(1 + 1/\rho)^2\right]$
- Inverse (approximate): given a desired mean $m$ and shape $\rho$, $\text{scale} = m / \Gamma(1 + 1/\rho)$

Notes:

- $\rho > 1$ → increasing hazard with age
- $\rho = 1$ → exponential model
- $\rho < 1$ → decreasing hazard

```yaml
phenotype_model1: frailty
phenotype_params1:
  distribution: weibull
  scale: 2160
  rho: 0.8
```

#### 1b. Exponential

$$
h(t) = \text{rate} \quad\text{(constant)}
$$

- Mean: $1/\text{rate}$
- Variance: $1/\text{rate}^2$
- Inverse: $\text{rate} = 1/\text{mean}$

```yaml
phenotype_model1: frailty
phenotype_params1:
  distribution: exponential
  rate: 0.02    # mean onset ~ 50
```

#### 1c. Gompertz

$$
h(t) = \text{rate} \cdot \exp(\gamma \cdot t)
$$

- Mean: $(1/\gamma) \cdot E_1(\text{rate}/\gamma) \cdot \exp(\text{rate}/\gamma)$, where $E_1$ is the exponential integral
- No simple closed-form inverse for $(\text{rate}, \gamma)$

Practical guidelines:

- $\gamma$ controls the speed of aging (0.03–0.10 realistic)
- `rate` sets the initial hazard
- To shift age of onset higher → decrease `rate`
- To make onset earlier → increase $\gamma$

```yaml
phenotype_model1: frailty
phenotype_params1:
  distribution: gompertz
  rate: 0.00005
  gamma: 0.07
```

#### 1d. Lognormal

$$
\log(\text{age at event}) \sim \mathcal{N}(\mu, \sigma^2)
$$

- Mean: $\exp(\mu + \sigma^2/2)$
- Variance: $(\exp(\sigma^2) - 1) \cdot \exp(2\mu + \sigma^2)$

Inverse — to get $\mu, \sigma$ from a desired age mean $m$ and standard deviation $s$:

$$
\sigma = \sqrt{\log\!\left(1 + (s/m)^2\right)}, \quad \mu = \log(m) - \tfrac{1}{2}\sigma^2
$$

```yaml
# Example: mean ~ 65, sd ~ 5
phenotype_model1: frailty
phenotype_params1:
  distribution: lognormal
  mu: 4.160
  sigma: 0.076
```

#### 1e. Loglogistic

$$
f(t) = \frac{(\text{shape}/\text{scale}) \cdot (t/\text{scale})^{\text{shape}-1}}{\left(1 + (t/\text{scale})^{\text{shape}}\right)^2}
$$

The hazard rises, peaks, then decreases.

- Median: $\text{scale}$
- Mean (only when shape > 1): $\text{scale} \cdot (\pi/\text{shape}) / \sin(\pi/\text{shape})$
- Inverse (shape > 1): given desired mean $m$, $\text{scale} = m \cdot (\text{shape}/\pi) \cdot \sin(\pi/\text{shape})$

Notes:

- shape > 1 → hazard increases then decreases
- shape = 1 → logistic distribution (monotone)

```yaml
phenotype_model1: frailty
phenotype_params1:
  distribution: loglogistic
  scale: 50
  shape: 3.5
```

#### 1f. Gamma (accelerated-time model)

$$
\text{age at event} \sim \text{Gamma}(\text{shape}, \text{scale})
$$

- Mean: $\text{shape} \cdot \text{scale}$
- Variance: $\text{shape} \cdot \text{scale}^2$

Exact inverse from desired mean and standard deviation:

$$
\text{shape} = (m/s)^2, \quad \text{scale} = s^2/m
$$

```yaml
# Example: mean ~ 60, sd ~ 10
phenotype_model1: frailty
phenotype_params1:
  distribution: gamma
  shape: 36.0       # (60/10)^2
  scale: 1.6667     # (10^2)/60
```

---

## 2. Mixture cure frailty model (`cure_frailty`)

A mixture cure model: the liability threshold determines **who** gets the
disorder; a proportional-hazards frailty model then determines **when**
(age-of-onset) among cases. Controls are censored at $10^6$.

Per trait:

$$
L_{\text{std}} = \mathrm{standardize}(L) \quad \mathcal{N}(0, 1)\text{ liability}
$$

$$
\text{is\_case} = L_{\text{std}} > \Phi^{-1}(1 - K) \quad\text{(top $K$ fraction)}
$$

$$
z = \exp(\beta \cdot L_{\text{std}}) \quad\text{(frailty, cases only)}
$$

$$
t_{\text{case}} = H_0^{-1}\!\left(\frac{-\log U}{z}\right), \quad t_{\text{ctrl}} = 10^6
$$

This separates case status (deterministic from liability rank) from
age-of-onset (stochastic from frailty + baseline hazard), unlike the pure
frailty model where both are entangled.

The `distribution` key in `phenotype_params` selects which baseline hazard
to use (any of the six distributions: weibull, exponential, gompertz,
lognormal, loglogistic, gamma).

**Config parameters**

| Key | Description |
|---|---|
| `prevalence1`, `prevalence2` | Population prevalence $K$ per trait |
| `beta1`, `beta2` | Liability effect on log-hazard (cases only) |
| `beta_sex1`, `beta_sex2` | Sex covariate effect (0 = no effect) |
| `phenotype_params1/2.distribution` | Baseline hazard distribution name |
| `phenotype_params1/2.<distribution params>` | Parameters for the chosen distribution |

**Example**

```yaml
phenotype_model1: cure_frailty
prevalence1: 0.10
beta1: 1.0
phenotype_params1:
  distribution: weibull
  scale: 2160
  rho: 0.8
```

---

## 3. ADuLT liability threshold model (`adult`, `method: ltm`)

From Pedersen et al., *Nat Commun* 2023.

A deterministic mapping from liability to age-of-onset via the logistic
cumulative-incidence-proportion (CIP) function:

$$
\mathrm{CIP}(\text{age}) = \frac{K}{1 + \exp\!\left(-k(\text{age} - x_0)\right)}
$$

where $K$ = prevalence, $x_0$ = midpoint age, $k$ = growth rate.

Steps:

1. Standardize $L$ to $\mathcal{N}(0, 1)$.
2. Case if $L > \Phi^{-1}(1 - K)$ (top $K$ fraction of liability).
3. Case age: $x_0 + \tfrac{1}{k} \cdot \log\!\left(\Phi(-L) / (K - \Phi(-L))\right)$ — i.e. CIP inverse applied to the individual's cumulative risk $\Phi(-L)$.
4. Controls: $t = 10^6$ (censored downstream).

Properties:

- Deterministic — the same liability always maps to the same age.
- Higher liability → younger onset age among cases.
- Case rate = prevalence (by construction).
- Case onset ages are centered around $x_0$.

Parameter $k$ controls the spread of onset ages:

- Small $k$ (e.g. 0.1): wide age-of-onset range (~10–90 years)
- Large $k$ (e.g. 0.5): concentrated near $x_0$ (~40–60 years)
- The scale parameter of the logistic distribution is $1/k$.

**Config parameters**

| Key | Description |
|---|---|
| `prevalence1`, `prevalence2` | Population prevalence $K$ per trait |
| `phenotype_params1/2.cip_x0` | Logistic CIP midpoint age (default 50) |
| `phenotype_params1/2.cip_k` | Logistic CIP growth rate (default 0.2) |

**Example**

```yaml
phenotype_model1: adult
phenotype_model2: adult
prevalence1: 0.10
prevalence2: 0.20
phenotype_params1:
  method: ltm
  cip_x0: 50
  cip_k: 0.2
phenotype_params2:
  method: ltm
  cip_x0: 50
  cip_k: 0.2
```

---

## 4. ADuLT proportional hazards model (`adult`, `method: cox`)

From Pedersen et al., *Nat Commun* 2023.

Weibull(shape = 2) proportional hazards with rank-based CIP-to-age mapping.

Steps:

1. Standardize $L$ to $\mathcal{N}(0, 1)$.
2. Raw event time: $t_{\text{raw}} = \sqrt{-\log U / \exp(L)}$, $U \sim \text{Uniform}(0, 1)$. (Weibull(shape = 2) frailty model with frailty $= \exp(L)$.)
3. Sort all individuals by $t_{\text{raw}}$ ascending.
4. Running CIP: $\mathrm{cip}_i = \mathrm{rank}_i / (N + 1)$.
5. Case if $\mathrm{cip}_i < K$ (prevalence).
6. Case age: $x_0 + \tfrac{1}{k} \cdot \log\!\left(\mathrm{cip}_i / (K - \mathrm{cip}_i)\right)$.
7. Controls: $t = 10^6$ (censored downstream).

Properties:

- Stochastic — the Weibull noise ($U$) introduces randomness.
- Liability affects case status probabilistically via the hazard.
- Case rate = prevalence (by rank cutoff).
- Case onset ages centered around $x_0$.

Key difference from `adult` (LTM):

- `adult` (LTM) is deterministic (liability alone determines case/age).
- `adult` (Cox) adds Weibull noise, so two individuals with the same liability may differ in case status and onset age.

**Config parameters** — same as the LTM variant above; the only distinguishing key is `method: cox`.

**Example**

```yaml
phenotype_model1: adult
phenotype_model2: adult
prevalence1: 0.10
prevalence2: 0.20
phenotype_params1:
  method: cox
  cip_x0: 50
  cip_k: 0.2
phenotype_params2:
  method: cox
  cip_x0: 50
  cip_k: 0.2
```

---

## 5. First-passage time model (`first_passage`)

Based on Lee & Whitmore (2006) and Aalen & Gjessing (2001).

A latent Wiener process $Y(t) = y_0 + \mathrm{drift} \cdot t + W(t)$
represents cumulative health degradation. Disease onset occurs at the
first time $Y(t) \le 0$. Event times follow an inverse Gaussian
distribution.

Unlike frailty models where the unique environment $E$ is a single draw at
birth, here $E$ operates as an ongoing stochastic process $W(t)$ —
Brownian motion that accumulates over the life course.

Per trait:

$$
L_{\text{std}} = \mathrm{standardize}(L) \quad \mathcal{N}(0, 1)\text{ liability}
$$

$$
y_{0,i} = \sqrt{\text{shape}} \cdot \exp(-\beta L_{\text{std}} - \beta_{\text{sex}} \cdot \text{sex}) \quad\text{(initial distance)}
$$

$$
Y_i(t) = y_{0,i} + \mathrm{drift} \cdot t + W_i(t) \quad\text{(Wiener process)}
$$

$$
T_i = \inf\{t : Y_i(t) \le 0\} \quad\text{(first passage)}
$$

Liability scales $y_0$ (initial distance to boundary):

- Higher liability → smaller $y_0$ → closer to boundary → earlier onset.
- $\beta > 0$ means higher liability is worse (same convention as frailty).
- $\beta = 0$ means liability has no effect on timing.

Two drift regimes:

- **drift < 0 (toward boundary)** — everyone eventually hits. Event times $\sim \mathrm{IG}(y_0/|\mathrm{drift}|, y_0^2)$. Mean event time = $\sqrt{\text{shape}} / |\mathrm{drift}|$.
- **drift > 0 (away from boundary)** — emergent cure fraction: $P(\text{never hit}) = 1 - \exp(-2 y_0 \cdot \mathrm{drift})$. Higher liability → smaller $y_0$ → **higher** probability of hitting and earlier onset among those who hit. Both susceptibility and timing emerge from a single $\beta$ parameter.
- **drift = 0** — degenerate case (Lévy-distributed FPT with infinite mean).

**Inverse Gaussian distribution.** $T \sim \mathrm{IG}(\mu, \lambda)$ where:

- $\mu = y_0/|\mathrm{drift}|$ (mean event time)
- $\lambda = y_0^2$ (shape parameter)

$$
f(t) = \sqrt{\frac{\lambda}{2\pi t^3}} \cdot \exp\!\left(-\frac{\lambda(t - \mu)^2}{2\mu^2 t}\right)
$$

$$
\mathbb{E}[T] = \mu, \quad \mathrm{Var}(T) = \mu^3/\lambda
$$

**Hazard shape.** The IG hazard has an "upside-down bathtub" shape:

- Starts at 0, rises to a peak, then decreases.
- Asymptotes to $\lambda/(2\mu^2) > 0$ — residual risk never vanishes.

This is distinct from Weibull (monotone only), Gompertz (monotone increasing), and lognormal (UBT but $h \to 0$).

Inverse — to get drift and shape from desired mean $m$ and CV:

$$
\text{shape} = m^2 / (m \cdot \mathrm{CV}^2), \quad \mathrm{drift} = -\sqrt{\text{shape}} / m
$$

(Since $\mathrm{Var} = \mu^3/\lambda$, $\mathrm{CV}^2 = \mu/\lambda$.)

**Config parameters**

| Key | Description |
|---|---|
| `beta1`, `beta2` | Liability effect on $\log(y_0)$ |
| `beta_sex1`, `beta_sex2` | Sex covariate effect on $\log(y_0)$ (0 = no effect) |
| `phenotype_params1/2.drift` | Wiener drift rate (negative = toward boundary, positive = cure) |
| `phenotype_params1/2.shape` | $y_0^2$ (initial distance squared; controls spread) |

**Examples**

Everyone hits, mean onset ~ 1000 time units:

```yaml
phenotype_model1: first_passage
beta1: 1.0
phenotype_params1:
  drift: -0.01
  shape: 100        # y0 = 10, mean = 10/0.01 = 1000
```

Emergent cure fraction:

```yaml
phenotype_model1: first_passage
beta1: 1.0
phenotype_params1:
  drift: 0.05
  shape: 100        # cure fraction ~ 1 - exp(-2*10*0.05) = 63.2%
```
