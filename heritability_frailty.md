# Heritability in a Frailty Model

## Two layers in the ACE simulation

ACE has two layers between genetics and the binary "affected" outcome:

1. **Liability layer**: `L = A + C + E` — a continuous normal variable with `h² = Var(A)/Var(L)`
2. **Event layer**: the Weibull frailty model converts liability into a *probability* of having the event by a given age

### Liability threshold model (what Falconer assumes)

Liability deterministically decides the outcome — you're affected if `L > cutoff`, period. Falconer's formula perfectly recovers the liability h².

```
L = A + C + E  ~  N(0, 1)
affected = I(L > threshold)
```

### Frailty model (what ACE simulates)

Liability only shifts the probability. A person with `L = 1.5` has a 27% chance of the event (trait 1), not 100%. Two siblings with identical genetics can have different outcomes by chance.

```
L = A + C + E  ~  N(0, 1)
hazard(t | L) = (rho/scale) * (t/scale)^(rho-1) * exp(beta * L)
P(event by t | L) = 1 - exp(-(t/scale)^rho * exp(beta * L))
```

## How liability maps to P(affected) at age 80

Using the default ACE parameters:

|   L   | Threshold (step) | Trait 1 (β=1.0, ρ=0.8) | Trait 2 (β=1.5, ρ=1.2) |
|------:|:----------------:|:----------------------:|:----------------------:|
| -2.00 |        0%        |         1.0%           |         0.9%           |
| -1.00 |        0%        |         2.6%           |         4.0%           |
|  0.00 |        0%        |         6.9%           |        16.5%           |
|  1.00 |        0%        |        17.7%           |        55.5%           |
|  1.28 |        0%        |        22.7%           |        70.8%           |
|  2.00 |      100%        |        41.1%           |        97.3%           |
|  3.00 |      100%        |        76.3%           |       100.0%           |

The frailty sigmoid is much more gradual than the threshold step function. This means the stochastic hazard process acts like extra environmental noise:

```
Effective model:  affected ≈ I(L > t + ε)
where ε is noise from the hazard process.

h²_effective = Var(A) / (Var(L) + Var(ε)) < Var(A) / Var(L)
```

## Quantifying the attenuation

From simulation with true h²_liability = 0.50 and PO pairs (r=0.5):

**Trait 1 (β=1.0, ρ=0.8):**
- Prevalence K = 0.103
- K among children of affected: 0.120
- corr(L_parent, L_child) = 0.268 (expected 0.25 = h²/2) — genetics are correct
- corr(affected_parent, affected_child) = 0.024 — severely attenuated
- Falconer h² (what EPIMIGHT estimates): **0.104**
- Ratio to true: **0.21×**

**Trait 2 (β=1.5, ρ=1.2):**
- Prevalence K = 0.268
- K among children of affected: 0.317
- corr(affected_parent, affected_child) = 0.082
- Falconer h² (what EPIMIGHT estimates): **0.234**
- Ratio to true: **0.47×**

## What controls the attenuation

| Factor             | Effect                                                |
|--------------------|-------------------------------------------------------|
| β (frailty weight) | Higher β → steeper sigmoid → less attenuation         |
| ρ (Weibull shape)  | Higher ρ → steeper hazard ramp → less attenuation     |
| K (prevalence)     | Higher K → more of the liability used → less attenuation |
| Follow-up time     | Longer follow-up → higher K → less attenuation        |

Sweeping β with Weibull scale=2160, ρ=0.8, t=80, h²_liability=0.50:

|   β  |   K    |  K_r   | Falconer h² | Ratio to true |
|-----:|-------:|-------:|------------:|--------------:|
|  0.5 |  0.077 |  0.080 |       0.022 |          0.04 |
|  1.0 |  0.103 |  0.119 |       0.100 |          0.20 |
|  1.5 |  0.143 |  0.184 |       0.207 |          0.41 |
|  2.0 |  0.188 |  0.246 |       0.278 |          0.56 |
|  3.0 |  0.261 |  0.341 |       0.369 |          0.74 |
|  5.0 |  0.343 |  0.435 |       0.447 |          0.89 |

At β→∞ the frailty model converges to a threshold model and Falconer recovers the full h².

## Deriving Var(ε_hazard)

Rewrite the frailty model as a latent threshold model using the inverse-CDF trick:

```
T ≤ t  ⟺  -log(U)  ≤  H₀(t) · exp(β·L)         U ~ Uniform(0,1)
       ⟺  β·L + W  >  -log(H₀(t))                W = -log(-log(U))
```

W follows a **standard Gumbel distribution**, independent of L. So the effective latent variable is:

```
L* = β·(A + C + E) + W
```

**Var(ε_hazard) = Var(W) = π²/6 ≈ 1.6449**

This is a constant — it does not depend on β, scale, ρ, or follow-up time. The effective heritability on the complementary log-log (cloglog) latent scale is:

```
h²_eff = β²·Var(A) / (β²·Var(L) + π²/6)
```

## Variance decomposition

### Trait 1 (β=1.0, h²_liability=0.50)

| Source                     | Variance | Share  |
|----------------------------|----------|--------|
| β²·Var(A) — genetic signal    |   0.500  | 18.9%  |
| β²·Var(C+E) — liability env   |   0.500  | 18.9%  |
| π²/6 — hazard stochasticity   |   1.645  | **62.2%** |
| **Total**                      | **2.645** |        |

The hazard process contributes the **majority** of the total variance at β=1.

### Trait 2 (β=1.5, h²_liability=0.50)

| Source                     | Variance | Share  |
|----------------------------|----------|--------|
| β²·Var(A) — genetic signal    |   1.125  | 28.9%  |
| β²·Var(C+E) — liability env   |   1.125  | 28.9%  |
| π²/6 — hazard stochasticity   |   1.645  | **42.2%** |
| **Total**                      | **3.895** |        |

## Formula vs Falconer (link function mismatch)

The cloglog formula and Falconer's estimate don't match exactly because Falconer assumes probit (normal) errors while the actual errors are Gumbel (skewed):

|   β  | h²_eff (cloglog) | Falconer (probit) | Gap    |
|-----:|-----------------:|------------------:|-------:|
|  0.5 |           0.0660 |            0.0218 | +0.044 |
|  1.0 |           0.1890 |            0.1018 | +0.087 |
|  1.5 |           0.2888 |            0.2333 | +0.056 |
|  2.0 |           0.3543 |            0.2756 | +0.079 |
|  3.0 |           0.4227 |            0.3713 | +0.051 |
|  5.0 |           0.4691 |            0.4426 | +0.027 |

The gap shrinks at higher β as the error distribution matters less relative to the signal.

## Analogies to other link functions

The Var(ε) depends on the link function used in the generalized linear model:

| Link              | Error distribution | Var(ε) |
|-------------------|--------------------|--------|
| Probit            | Normal             | 1      |
| Logit             | Logistic           | π²/3 ≈ 3.29 |
| Complementary log-log (frailty) | Gumbel  | **π²/6 ≈ 1.64** |

The frailty model sits between probit and logit in terms of added noise.

## References

- Falconer DS (1965). The inheritance of liability to certain diseases, estimated from the incidence among relatives. *Ann Hum Genet* 29:51-76.
- Dempster ER, Lerner IM (1950). Heritability of threshold characters. *Genetics* 35:212-236.
