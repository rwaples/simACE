# Transmitted shared environment: literature against ADR 0025

Written 2026-10-14 for plan v3 of vertical C transmission
(`plans/vertical-c-transmission-v3.md`, session-local draft; ADR
`docs/adr/0025-vertical-c-transmission.md`, Proposed, on the `worktree-vertical-c-transmission` branch and not yet on `dev`; glossary entries
"Common environment component", "C transmission", "Household" in `GLOSSARY.md`).
The model under check is $C_{child} = b_m C_{mother} + b_f C_{father} + u_{household}$
with the configured $C$ as the total variance each generation.

How this note was made. Five parallel searches (path-analytic cultural
inheritance; extended twin-family designs; maternal effects and animal breeding;
genetic nurture and SEM-PGS; registry and liability-threshold work), each reading
primary sources from PMC, Europe PMC, publisher OA, bioRxiv/PsyArXiv, HAL or
author copies, plus my own reading of Herzig et al. 2026 (local PDF in
`external/pdfs/`, its supplement and the authors' R code). Every bibliographic
field below was re-checked on 2026-10-14 against the Crossref API
(`api.crossref.org/works/<doi>`) or, for the AJHG papers without DOIs, PubMed
esummary. Equations cited with a page or equation number were read from the
rendered page, not from `pdftotext`. The load-bearing ones (Cloninger II eqs 10.2,
15–20, 29–33; Rao–Morton–Cloninger eqs 1–8, 23; Herzig main-text Table 2 and
supplement eqs 27–31, 93) I re-read myself. Anything marked **derived** is algebra
from a paper's printed equations, not a statement in the paper. Anything marked
UNVERIFIED was not confirmed against a rendered page.

Numeric check, independent of simACE code: a 2 M-couple toy recursion
(`shared_c2.py`, session-local script, not in the repo; random mating, $A=.5, C=.3, E=.2$,
$b_m=.5, b_f=.3$) reproduced every ADR fraction to three decimals (MO .500, FO
.301, FS 1.001, MHS .911, PHS .089, 1C sisters .250, brothers .090, mixed .151,
maternal grandmother .250, paternal grandfather .090, aunt .501, uncle .300,
Var(C) 1.001), and under rank-matched phenotypic AM at $r=.4$ with $b_m=.5$ gave
Cov(A,C) = .0152 against .0150 expected and corr(C_father, C_child) = .0606
against .0600. The MHS value .911 holds only because the toy shares one $u$
across a mother's matings; with a fresh $u$ per mating it is .250 ($= b_m^2$).

## 1. Taxonomy

Classes: (1) environment → environment; (2) parental phenotype → offspring
environment; (3) parental genotype → offspring environment (genetic nurture);
(4) maternal-effect coefficient on the maternal phenotype; (5) co-residence
environment shared by parents and children. "Stationarity" says whether the
transmitted environment's variance is held at a target (fresh term tops up) or
whether the fresh term is fixed and the total settles or grows.

| Paper | Class | Transmitted quantity | Stationarity | AM handling | Relationship covariances given |
|---|---|---|---|---|---|
| Rice, Cloninger & Reich 1978 [RCR78] | 1 (sex-specific $\beta_M,\beta_F$) | parental cultural value $B$ | RM: fresh term $=(1-\beta_M^2-\beta_F^2)V_B$ (eq 19); AM: fresh fixed at RM value, $V_B$ inflates (eq 3.1, 20) | primary phenotypic, $m$ | mates, PO (sex-specific), FS, GP, vertical-$n$, avuncular, cousins, collateral-$n$ (eqs 5–7, 47–52); half-sibs absent |
| Cloninger, Rice & Reich 1979 II [CRR79] | 1 | $B$, per-child residual $F$ | as RCR78 (eq 5 RM, eq 7 AM); bound $\beta_M^2+\beta_F^2+2m\beta_M\beta_F\le1$ (eq 3) | primary phenotypic; induced $w$ (eqs 15–19) | PO (10.2), FS (20), DZ/MZ (21–22), GP (23–25), vertical (26–28), avuncular (29), cousins (31), collateral (33), adoptive/foster (Table 3) |
| Cloninger, Rice & Reich 1979 III [CRR79b] | 1 + rearing arrangements | $B$ per rearing adult | as above | as above | half-sibs under four rearing arrangements (eq 20, not transcribed) |
| Rao, Morton & Cloninger 1979 [RMC79] | 1 + shared sibship term (class 5 for sibs) | parental indexed environment $C$ via $f_F, f_M$; shared non-transmitted $B$ | standardized each generation ("assuming equilibrium") | mixed homogamy ($p$ phenotypic + $m,u$ social) | PO (3–5, Table 2), sibs (Table 2), vertical-$n$ (16–17), aunt–niece (18), cousins (19), half-sibs and MZ-twin kin (Table 3, 21–22) |
| Rao, Morton & Yee 1976 [RMY76] | 1 + 2 | parental childhood environment ($f$) and adult phenotype ($x$) | standardized ("assuming constancy over generations", p 231) | social homogamy $m,s,u$ | nuclear, adoptive, half-sibs (HS together $u^*=1$ by assumption, p 234), cousins (Table 4; footnotes illegible) |
| Rao, Morton & Yee 1974 [RMY74] | 5 | none transmitted | n/a | $m$ | MZ, FS, HS, PO, adoptive (pp 337–345); states that extending beyond 2nd degree "requires additional assumptions about transmission" (p 346) |
| Cavalli-Sforza & Feldman 1973 [CSF73] | 2 (phenotype → phenotype, genotype-specific plasticity) | midparent phenotype, $2b_i$ | fresh $\sigma^2$ fixed; variance accumulates to an eigenvalue-bounded equilibrium (eqs 26–30) | none worked | PO, foster PO, MZ apart (eqs 34–43) |
| Eaves, Last, Young & Martin 1978 [ELYM78] | 2 | parental phenotype → offspring environment | G–E correlation stationarity $s'=s$ treated as a testable hypothesis (p 296–298) | phenotypic | Table 17 (not transcribed) |
| Keller et al. 2009 Cascade [K09] | 2 (class 1 named as a variant, p 14–15) | whole parental phenotype, path $m$ | equilibrium assumed; transmitted $V_F$ derived, sibling $V_S$ free (eq 3, Table 1) | primary phenotypic copath $\mu$; $w$ by eq 6 | 17 relative classes incl. GP, avuncular, cousins via twins (App. A); no half-sibs |
| Keller, Medland & Duncan 2010 [K10]; Medland & Keller 2009 [MK09]; Maes et al. 2009 [M09] | 2 | as K09 | as K09 | as K09 | simulation bias and sub-design identifiability only |
| Coventry & Keller 2005 [CK05]; Keller & Coventry 2005 [KC05] | 2 | as K09 | as K09 | as K09 | CTD bias formulae (KC05 eqs 9–14) |
| Eaves et al. 1999 [E99]; Truett et al. 1994 [T94, paywalled] | 2 | parental phenotype, four sex-specific paths | equilibrium, unit variance constraint (p 72) | phenotypic + social homogamy | 80 correlations (formulae deferred to T94) |
| Maes et al. 2006 [M06]; Maes et al. 2023 [M23] | 2, with a class-1 "C to C" alternative fitted (M06 p 806) | parental phenotype (or $C$) | equilibrium constraints | phenotypic vs social homogamy | twins, parents, sibs, spouses; liability-threshold (binary) in both; Swedish registers in M23 |
| Herzig, Noûs, Saint Pierre & Perdry 2026 [H26] | 1 | parental environment $E$, correlation $\nu$ | $e$ fixed; residual set from the realized mates' $E$ correlation (code `pop.sim.r`; supp. eq 27–28) | primary phenotypic $r_{ho}$; induced $\rho$ by supp. eq 31 | trio only: mates, PO, $A$–$E$ (main Table 1–2); midparent regression (supp. eq 93) |
| Lande & Kirkpatrick 1990 [LK90]; Kirkpatrick & Lande 1989 [KL89, paywalled] | 2/4 (+5 fresh family term) | maternal (and paternal, App. 1) phenotype, matrix $M$ | fresh fixed; stable iff $\lvert\lambda(M)\rvert<1$ (p 192) | random mating | midparent (p 190), mother–daughter (p 192), FS and paternal HS (App. 3) |
| Riska, Rutledge & Atchley 1985 [RRA85] | 3 + maternal-performance chain $m_m$ | maternal performance $M$ | fresh fixed; series in $m_m/2$ converges for $\lvert m_m\rvert<2$ (p 296) | random mating | breeding-value–phenotype only |
| Koerhuis & Thompson 1997 [KT97] | 3 (+5 dam permanent environment $c^2$; Falconer $F_m$ model in App.) | dam phenotype ($F_m$) or dam environment ($\sigma_{EC}$, one step) | fresh fixed; $\sigma_P^2=[\ldots]/(1-F_m^2)$ (p 247) | random mating | dam–offspring (p 246); FS/MHS via $c^2$ (p 230–231) |
| Varona et al. 2015 [V15] | 3 (Willham) | maternal genotype | none | random mating | FS, MHS, DO (Methods); PHS, SO derivable |
| Balbona, Kim & Keller 2021 [BKK21]; Kim, Balbona & Keller 2021 [KBK21] | 2 (+3 measurement) | parental phenotype, $F_o=fY_p+fY_m$ | $f$ fixed; $V_F=2f^2V_Y(1+\mu V_Y)$ floats to equilibrium (eqs 5, 16) | primary phenotypic copath $\mu$ | trio only (ESM Table I) |
| Kong et al. 2018 [K18]; Young et al. 2018 [Y18] | 3 | parental genotype | n/a | one generation (K18) / none (Y18) | Y18 eq 2: $v_gR+v_{e\sim g}R_{par}+c_{g,e}R_{o,par}$; K18 PO/sib formulae in blocked supplement |
| Nivard et al. 2024 [N24] | 3, extended family | parental and avuncular PGI | n/a | acknowledged, no formula | empirical avuncular effect |
| Lichtenstein et al. 2009 [L09]; Pettersson et al. 2019 [P19]; Sandin 2014/2017 [S14, S17]; Kendler 2015/2016/2018/2020 [Ke15, Ke16, Ke18, Ke20]; Xia et al. 2016 [X16]; Krebs et al. 2024 [Kr24] | 5 | none transmitted | fresh per family | noted as a confound (X16, Ke15) | fixed fractions: L09 Table 1 FS 1, MHS 1, PHS 0, PO 0 (+ adult spouse term); X16 PO = FS = couple = 1; Ke16 co-residence-year weights |
| Falconer 1965/1967 [F65, F67]; Wray & Gottesman 2012 [WG12]; Athanasiadis 2022 [A22]; Zaitlen 2013 [Z13] | none | none | n/a | n/a | $h^2=b/r$ estimators; empirical per-relationship gradients |

## 2. The closest published analogues

**Rao, Morton & Cloninger 1979 is the structural twin of ADR 0025.** Their Fig. 2
(p 179) has $C_1 \leftarrow f_F C_F + f_M C_M + b\,B$ with $B$ "non-transmitted
common environment for a sibship" shared by the two children, separate maternal
and paternal paths "to retain specific maternal effects" (p 178), and an explicit
rejection of parental-phenotype paths ("Direct effects of parental phenotypes on
the environment of a child are assumed negligible since they were not significant
even for IQ", p 178). Sib environment correlation is $b^2+\psi$ with

$$\psi = f_F^2 + f_M^2 + 2 f_F f_M (u + p\phi^2) \qquad \text{(eq 2, p 179)}$$

and mother–child environment correlation $\rho_{C_M C_1} = f_M + f_F(u + p\phi^2)$
(eq 6, p 180). simACE is the special case $b^2 = 1 - \psi$, which makes the FS
fraction exactly 1. Their section 8 (eq 23, p 186) is the minimum-$\chi^2$ over
$z$-transformed per-relationship correlations with $\sigma^2=1/(n-3)$ that the
later EPIMIGHT GLS resembles; they note it "ignores correlations between
correlations" and cite evidence that this changes little.

**The Washington series (RCR78, CRR79, CRR79b) is the same class with a
per-child residual.** CRR79 Table 1 (p 177) defines $B$ "Sociocultural factors
transmitted between generations" and $F$ "Within family variation of
transmissible sociocultural values". Their FS and MZ environment fraction is
therefore $\beta_M^2+\beta_F^2+2\beta_M\beta_F r_{BB}$, not 1 (CRR79 eqs 20, 22,
p 183), and a separate sib correlation $c_s$ on the non-transmitted $E$ carries
the rest. CRR79 p 196 argues against classes 2 and phenotype-to-phenotype as
"neo-Lamarckian": "the As and Bs of parents and children are correlated, and the
Es are independent." That is the ADR's argument for rejecting option (c).

**Herzig et al. 2026 is the only paper found whose simulation uses simACE's
top-up convention.** Supplement §3.4 (p 10) fixes $\mathrm{var}(E_i)=e^2$ for
parents and child and sets the trio covariance matrix (eq 27) with
parent–offspring environment correlation $\nu$; the authors' `pop.sim.r` draws
the offspring environment as

```r
cpm   <- cor(Ep, Em)                      # realized mates' E correlation
cx    <- nu/(1 + cpm)                     # coefficient on each parent
sdres <- sqrt(1 - 2*nu**2/(1 + cpm))*e    # residual tops up to e^2
E1    <- cx*(Ep + Em) + rnorm(pop.size/2, sd = sdres)
```

(github.com/HervePerdry/AMVCTpaper, `R/pop.sim.r`, read 2026-10-14). The residual
variance is computed from the *realized* cross-mate correlation each generation,
which is ADR decision 3 ("realized rather than analytic") in a published model.
Differences: their $\nu$ is the parent–offspring correlation, so the coefficient
is $b=\nu/(1+r_{E_1E_2})$ under AM; their residual is per offspring (sibs share
only the transmitted part); the transmitted quantity is the whole environment
$E$, not a $C$ distinct from $E$. The feasibility bound is supp. eq 28:
$\nu \le \sqrt{(1+r_{E_1E_2})/2}$, which in coefficient form is
$2b^2(1+r_{E_1E_2}) \le 1$.

**Maes et al. 2006 is the only fit of a class-1 model found.** For smoking
initiation on the Virginia 30,000 (liability threshold), with twins and parents,
"C to C transmission was not significant" and "the social homogamy model fitted
almost as well as the phenotypic assortment model" (p 806). Keller et al. 2009
(p 14–15) describe class 1 as a Cascade variant in which "parents pass only
certain (e.g., environmental) aspects of their phenotype to offspring" but give
no algebra for it, and Keller et al. 2010 state "we did not investigate
alternative modes of vertical transmission".

**Animal breeding has no class-1 recursion.** The fresh, non-inherited dam term is
the "maternal permanent environmental effect" $c^2$ (KT97 p 230–231; the
$b_m=b_f=0$ case). The one-step environmental dam–offspring covariance
$\sigma_{EC}$ (KT97 p 231, citing Meyer 1992b) is the nearest thing to $b_m C$,
and a recursion on the maternal phenotype is a "cascading maternal effect"
(McGlothlin & Galloway 2014 supplement [MG14]; Kuijper & Hoyle 2015 [KH15]) or "a cycle of
maternal effects of length one" (LK90 p 192).

## 3. Verdicts on the design checks

### 3.1 Random-mating shared-C fractions

Claim: MZ/FS 1, MO $b_m$, FO $b_f$, MHS $1-b_f^2$, PHS $b_f^2$, cousins
$b_m^2$ / $b_f^2$ / $b_m b_f$ by lineage.

- **MO $b_m$, FO $b_f$: confirmed.** RCR78 p 628: "In the random mating
  population, the father-offspring correlation is equal to $\beta_F b_o^2$, and
  the mother-offspring correlation is $\beta_M b_o^2$." CRR79 eq 10.2 at $m=0$:
  $\rho_{io} = \tfrac12\Gamma + \beta_i\Phi$ with $\Gamma=h^2$, $\Phi=b^2$ at
  $w=0$. RMC79 eq 6 at $u=p=0$: $\rho_{C_MC_1}=f_M$.
- **FS = MZ = 1: confirmed as the RMC79 special case** $b^2 = 1-f_F^2-f_M^2$
  (eq 2 and Table 2). It is *not* what the Washington papers give
  ($\beta_M^2+\beta_F^2$ plus a separate $c_s$), because their residual is per
  child. The ADR should say which convention it follows and why (household = one
  draw per mother).
- **Grandparent, avuncular, cousins: confirmed as lineage products.** CRR79
  eq 25 at $m=0$ gives grandparent $\beta_i\beta_k b^2$; eqs 29 and 31 give
  avuncular $(\beta_M^2+\beta_F^2)\beta_k b^2$ and cousins
  $(\beta_M^2+\beta_F^2)\beta_k\beta_{k'}b^2$, where the sib-sharing factor
  $\beta_M^2+\beta_F^2$ becomes 1 under simACE's shared $u$. RMC79 eqs 16–19
  carry sex-labelled $K_n,\Lambda_n$ recursions ("Sex-labelling of every parent
  is important in the presence of maternal effects", p 183), and cousins pick up
  $(b^2x^2+\psi)\Lambda_2\Lambda_2'$ (eq 19), which is $b_m^2$, $b_f^2$ or
  $b_m b_f$ with $x=1$ and $b^2=1-\psi$.
- **MHS $1-b_f^2$ and PHS $b_f^2$: no published counterpart; design-specific.**
  No paper prints half-sib fractions for a class-1 model with a shared residual.
  CRR79b makes half-sibs rearing-arrangement specific (eq 20, four arrangements,
  p 377) and warns that "if this is ignored, 'maternal effects' would be falsely
  attributed to the influence of the mother" (p 378). RMY76 sets half-sibs
  reared together to $u^*=1$ by assumption (p 234). L09 Table 1 sets C = 1 for
  maternal half-sibs and 0 for paternal. The value $1-b_f^2$ follows only from
  simACE's rule that the fresh draw is shared across a mother's matings; with a
  draw per mating the MHS fraction is $b_m^2$ (toy check above). Say so in the
  ADR, because the planned FS/MHS/PHS contrasts hinge on it (see §4).
- **Maternal-line asymmetry in general: consistent with the maternal-effect
  literature.** Willham-type covariances put the maternal term in FS, MHS and
  dam–offspring but not PHS or sire–offspring (V15 Methods: $cov_{FS}=\tfrac12
  s_a^2+s_{am}+s_m^2$, $cov_{MHS}=\tfrac14 s_a^2+s_{am}+s_m^2$,
  $cov_{DO}=\tfrac12 s_a^2+\tfrac54 s_{am}+\tfrac12 s_m^2$); LK90 App. 3 gives
  paternal half-sib $t=h^2/4$ and full-sib $t=h^2/2+c^2+m^2+mh^2/(1-m/2)$.
  Class-3 nurture, by contrast, is symmetric in the two half-sib types (Y18 eq 2,
  $R_{par}=\tfrac12$ for both).

### 3.2 AM-induced A–C covariance (one generation of maternal transmission)

Claim: $\mathrm{Cov}(A_{child},C_{child}) = \tfrac12 b_m r A C/V_P$ and
$\mathrm{corr}(C_{father},C_{child}) = b_m r C/V_P$.

**Confirmed by CRR79 eqs 15–18 (p 182), read from the page.** With standardized
$A$ and $B$, $r_{AB}=m\gamma\phi$ (eq 17), $\gamma=h+wb$, $\phi=b+wh$, and

$$w' = \tfrac12(\beta_M+\beta_F)(w + r_{AB}) \qquad \text{(eq 18)}.$$

**Derived.** With $\beta_F=0$ and parents' $w=0$ (so $\gamma=h$, $\phi=b$),
$w'=\tfrac12\beta_M m h b$ and $\mathrm{Cov}(A,B)=w'\,h b\,V_P = \tfrac12 b_m r
(A/V_P)(C/V_P)V_P = \tfrac12 b_m r A C/V_P$. The father's cultural term in eq 10.2
is $(\beta_F + m\beta_M)\Phi$, i.e. $\mathrm{Cov}(B_F,B_o)=\beta_M r_{BB}$ with
$r_{BB}=m\phi^2$ (eq 16), giving $b_m r C/V_P$ in simACE units. Keller et al.
2009 eq 6 ($w=(qa+wf)m+(qa+wf)\mu\sigma^2 m$, $\mu=r/V_P$) and Balbona et al.
2021 ESM Table I (cov$(Y_p,F_m)=V_Y\mu\,$cov$(Y_m,F_m)$) are the class-2 copath
rules and give the same result when applied to class 1 (both sub-agents derived
it independently). H26 supp. eq 31 is the same recursion in their notation.

**Extension the ADR does not state: the one-generation value is not the
equilibrium.** CRR79 p 183: at equilibrium $w=\lambda r_{AB}=\lambda m\gamma\phi$
with $\lambda=(\beta_M+\beta_F)/(2-\beta_M-\beta_F)$ (eq 18.1), the root of the
quadratic eq 19; RMC79 eq 1 has the same $\lambda$. H26 supp. eq 31 gives the
per-generation recursion and Fig. 2 shows $\rho$ rising over roughly 10–15
generations. For $b_m=.5$, $b_f=0$, $\lambda=1/3$ versus the one-generation
factor $1/4$, and $\gamma\phi$ itself grows with $w$. Since simACE runs many
generations, the realized Cov(A,C) in late generations will exceed the toy value,
so the ADR's formula should be labelled "first generation after transmission
starts" and the truth source (decision 4) remains the right reference.

### 3.3 Midparent–offspring slope under random mating

Claim: $[A + (b_m+b_f)C]/V_P$.

**Confirmed, by two routes, neither printed as such.** (i) **Derived** from CRR79
eq 10.2 at $m=0$: $\rho_{Mo}+\rho_{Fo} = h^2 + (\beta_M+\beta_F)b^2$, and the
midparent regression equals the sum of the two single-parent correlations when
mates are uncorrelated (the registry sub-agent obtained the same from Falconer
1965 eq 3 applied to MO and FO). (ii) H26 supp. §7.2 eq 93 (p 29–30):

$$h^2_{MP} = h^2 + \frac{(3+r_{ho})\rho a e + 2\nu e^2}{(1+r_{ho})\sigma^2},$$

which at $r_{ho}=0$, $\rho=0$ is $(a^2+2\nu e^2)/\sigma^2$; with symmetric
$b_m=b_f=\nu$ and $e^2\mapsto C$ that is the claim. Under AM, eq 93 shows the
slope picks up both the $A$–$E$ covariance and a $(1+r_{ho})$ denominator; the
ADR's gate (analytic line only without AM) is therefore right, and eq 93 is a
candidate reference for the AM case once $E$ is mapped to $C$ (only $C$ is
transmitted in simACE, so the mapping is not exact: UNVERIFIED).

### 3.4 Is "configured C is total; the fresh draw tops up" established?

**Under random mating, yes, universally in the path-analytic literature, but
unnamed.** RCR78 eq 2.2 (p 622): $f'\sigma_{F'} = (1-2\beta'^2)^{1/2}\sigma_{B'}$;
eq 19 (p 627): residual factor $(1-\beta_M^2-\beta_F^2)^{1/2}$; CRR79 eq 5
(p 180): $f^2V_F = (1-\beta_M^2-\beta_F^2)V_{B_o}$, "the cultural variance due to
fluctuation (F) from midparent cultural value"; RMC79 and RMY76 standardize every
variable each generation ("assuming constancy over generations", RMY76 p 231).
Names for the fresh term: "fluctuation about midparent cultural value" (RCR78
p 621), "within family variation of transmissible sociocultural values" (CRR79
Table 1), "non-transmitted common environment for a sibship" (RMC79 Fig. 2
legend), "sibling environment" $S$ (K09 Table 1), "residual common environment"
(CK05 p 215).

**Under assortative mating the literature splits, and simACE sits with Herzig.**
The Washington papers hold the fresh variance at its random-mating value and let
$V_B$ inflate: RCR78 p 621 "Once assortative mating occurs, the variance is
increased in successive generations until equilibrium is reached ... the partial
regression coefficient $f'$ is unaffected by assortative mating"; CRR79 eq 7
$V_B = V_{B_o}(1-\beta_M^2-\beta_F^2)/(1-(\beta_M^2+\beta_F^2+2\beta_M\beta_F
r_{BB}))$. Keller/Balbona likewise fix the coefficient and let $V_F$ float
(BKK21 eq 16; KBK21 "variances ... could increase over their base population
values as a consequence of VT or AM"). H26 holds $e$ fixed and shrinks the
residual by the realized mates' correlation, as simACE does. CSF73 (class 2) fixes
the residual and lets variance accumulate even under random mating (eqs 26–30),
which is the "additive variance" option the ADR rejected. The maternal-effects
literature is all fixed-fresh with a closed-form stationary total (KT97 p 247
$\sigma_P^2 = [\ldots]/(1-F_m^2)$; LK90 $\lvert\lambda(M)\rvert<1$).

**Pitfall for the feasibility bound.** Because of the mates' $C$ correlation, the
exact bound is CRR79 eq 3, $\beta_M^2+\beta_F^2+2m\beta_M\beta_F \le 1$ in their
standardized units, or H26 eq 28. In simACE units with $r_{CC}\approx r\,C/V_P$
(CRR79 eq 16 at $w=0$), the transmitted variance is
$(b_m^2 + b_f^2 + 2 b_m b_f r_{CC})\,C$, so the config-load check
$b_m^2+b_f^2\le1$ is necessary but not sufficient under positive AM with both
coefficients nonzero. The ADR already says the bound is "necessary under random
or positive assortative mating"; it could add the analytic AM form as a
config-load *warning* and leave the run-time sampling check as the arbiter.

## 4. Identifiability and bias, for the later GLS estimator

- **Sib-only sets cannot separate $C$ from $b_f$.** Derived: under the model
  $r_{FS}=\tfrac12 A + C$, $r_{MHS}=\tfrac14 A+(1-b_f^2)C$,
  $r_{PHS}=\tfrac14 A + b_f^2 C$, so $r_{MHS}+r_{PHS}=r_{FS}$ identically. The
  three sib correlations carry two pieces of information. With lineage-split PO
  ($r_{MO}=\tfrac12 A+b_mC$, $r_{FO}=\tfrac12 A+b_fC$) the four unknowns
  $A,C,b_m,b_f$ are generically identified from FS, PHS, MO, FO (nonlinear in
  $b_f$); cousins add lineage products only if pedigree-graph splits them by
  lineage, which it does not (ADR decision 4). $r_{MHS}+r_{PHS}=r_{FS}$ is also a
  testable restriction of the household-by-mother rule.
- **Classical warnings transfer directly.** CRR79 p 191–192: with $\beta=\tfrac12$
  every intact-family correlation is a function of $t^2=h^2+b^2+2whb$ only;
  "even with $\beta\ne\tfrac12$ the independence of $h$ and $b$ is weak";
  separation data "nearly essential". RCR78 p 633: "Information about
  first-degree relatives and mates in one population is insufficient to estimate
  $\tau$." K10 Table 3: $V_F$ is identified by $CV(PO)-CV(\text{MZ avuncular})$
  and $CV(\text{DZ cousin})-CV(\text{MZ cousin})$; $V_S$ by $CV(DZ)-CV(PO)$;
  $V_A$, $V_D$, $V_F$ collinear (VIF 32.6). M09: no two-generation sub-design
  identifies $D$ and $F$ together. KT97 p 226, 240–242 (citing Meyer 1992b,
  Robinson 1994, journal version [R96]): a maternal-line transmission is absorbed as negative
  $\sigma_{AM}$ plus $c^2$ when data are FS/MHS/DO-heavy; V15: FS/MHS/DO alone
  cannot even distinguish a maternal from a paternal-gametic model, sire–offspring
  and PHS-vs-MHS contrasts do.
- **Bias when transmission is ignored (derived from Falconer 1965 eq 3,
  $h^2=b/r$, random mating):** MO $(A+2b_mC)/V_P$; FO $(A+2b_fC)/V_P$; FS
  $(A+2C)/V_P$; MHS $(A+4(1-b_f^2)C)/V_P$; PHS $(A+4b_f^2C)/V_P$; cousins
  $(A+8b_xb_yC)/V_P$. The FS-vs-MHS design of P19 returns $A+4b_f^2C$. The
  per-relationship estimates are not monotone in degree, which matches the
  empirical gradients Z13 reports (PO > avuncular; half-sibs > grandparent by
  .125; cousins > avuncular) and Z13's open question "do parent/child pairs have
  the same environmental sharing as siblings?". Printed analogues: RCR78 Table 1
  (stature: fixing $\tau=\tfrac12$ gives $t_o^2=.728$ versus $.484$ with $\tau$
  free); CRR79 eq 10.4 (polygenic PO fit returns $t^2$); KC05 eqs 9–14 and CK05
  (CTD $V_A$ +63%, $V_C$ over by AM and induced $CV(A,F)$, under by $V_{NA}$);
  Y18 Table 1 (pedigree h² 92.7% vs true 40% under class-3 nurture).
- **The classical twin design is unaffected by C transmission alone.** Derived:
  MZ and FS both carry the full $C$, so $2(r_{MZ}-r_{FS})=A/V_P$; with AM the
  induced $w$ enters MZ and DZ equally and cancels, leaving the AM-only bias
  $A(1-r_{AA})/V_P$ that simACE's validator already uses. Kong et al. 2018 make
  the same point for parental genetic nurture ("the effects of parental genetic
  nurture are cancelled out"), with the caveat that sibling nurture does not
  cancel.

## 5. Recommendations bearing on plan v3

1. **Terminology.** Keep "C transmission" as the glossary term; note that the
   literature calls class 1 "cultural transmission" or "cultural inheritance" too
   (RCR78, CRR79, RMC79), so the glossary's reservation of "cultural transmission"
   for phenotype → environment is a local convention and should say so. For the
   fresh draw, cite RMC79's "non-transmitted common environment for a sibship"
   as the published name; "maternal effect" stays reserved (Willham/Falconer
   meaning). Nobody calls the top-up a top-up; "configured C is the total
   variance, held stationary" is accurate.
2. **Cite precedent in the ADR.** Random-mating top-up: RCR78 eq 19 / CRR79
   eq 5. Realized-correlation top-up under AM: H26 `pop.sim.r` and supp. eq 28.
   Structural analogue: RMC79 Fig. 2 and eqs 2, 6. Class-1 fit: M06 p 806.
   Rejection of option (c): CRR79 p 196.
3. **Label the AM formula as first-generation** and add the equilibrium form
   $w=\lambda m\gamma\phi$, $\lambda=(b_m+b_f)/(2-b_m-b_f)$ (CRR79 eq 18.1), or
   the per-generation recursion (CRR79 eq 18 / H26 supp. eq 31), as the reference
   for the informational AM check.
4. **State the half-sib caveat.** MHS $1-b_f^2$ and PHS $b_f^2$ are consequences
   of household = mother sharing one $u$ across matings; they have no literature
   counterpart and $r_{MHS}+r_{PHS}=r_{FS}$ follows. Record the per-mating
   alternative ($b_m^2$) as a considered option.
5. **Feasibility bound.** Add the AM-aware form
   $b_m^2+b_f^2+2b_mb_f\,r_{CC}\le1$ with $r_{CC}\approx r\,C/V_P$ as a warning
   at config load when AM is on (CRR79 eq 3; H26 eq 28).
6. **Midparent slope under AM.** H26 supp. eq 93 is the published form for the
   symmetric case; it supports keeping the analytic line off under AM.
7. **For the estimator workstream.** RMC79 eq 23 is the $z$-transform
   minimum-$\chi^2$ precedent; the sib-only collinearity in §4 means lineage-split
   PO (or lineage-split cousins) is required to identify $b_m$ and $b_f$; Kemper
   et al. 2021 (Nat Commun 12:1050, verified, not read) is the modern
   whole-relatedness-spectrum regression and should be read by whoever owns that
   workstream.

## 6. Paywalled or not retrieved (DOI, what is needed)

- Feldman MW, Cavalli-Sforza LL. Aspects of variance and covariance analysis
  with cultural inheritance. Theor Popul Biol 1979;15(3):276–307.
  doi:10.1016/0040-5809(79)90043-1. Need: the transmitted-environment variance
  recursions with and without AM, PO and sib covariances, identifiability
  remarks (equation numbers unknown).
- Cavalli-Sforza LL, Feldman MW. The evolution of continuous variation. III.
  Joint transmission of genotype, phenotype and environment. Genetics
  1978;90(2):391–425. doi:10.1093/genetics/90.2.391 (PMC1213897; free but
  bot-blocked from this host). Need: the joint class-1+2 iteration equations.
- Truett KR et al. A model system for analysis of family resemblance in extended
  kinships of twins. Behav Genet 1994;24(1):35–49. doi:10.1007/BF01067927. Need:
  pp 35–49, the GE-covariance equilibrium constraint and the 80-relationship
  expectation table.
- Heath AC, Kendler KS, Eaves LJ, Markell D. The resolution of cultural and
  biological inheritance: informativeness of different relationships. Behav Genet
  1985;15(5):439–465. doi:10.1007/BF01066238. Need: the informativeness tables by
  design.
- Willham RL. Biometrics 1963;19(1):18–27, doi:10.2307/2527570; Willham RL.
  J Anim Sci 1972;35(6):1288–1293, doi:10.2527/jas1972.3561288x. Need: the
  covariance-by-relationship tables and the grandmaternal cascade term.
- Kirkpatrick M, Lande R. The evolution of maternal characters. Evolution
  1989;43(3):485–503. doi:10.1111/j.1558-5646.1989.tb04247.x. Need: equilibrium
  $P$, mother/father–offspring and sib covariances.
- Thompson R. The estimation of maternal genetic variances. Biometrics
  1976;32(4):903–917. doi:10.2307/2529273. Need: Falconer-model expectations by
  relationship.
- Meyer K. Bias and sampling covariances of estimates of variance components
  due to maternal effects. Genet Sel Evol 1992;24(6):487–509.
  doi:10.1186/1297-9686-24-6-487 (nominally OA; every route failed). Need: the
  $\sigma_{AM}$/$\sigma_{EC}$ sampling-correlation table.
- Kong et al. 2018 Science supplementary material (doi:10.1126/science.aan6877):
  the induced PO and sibling correlations under genetic nurture and the AM
  derivation. Blocked by a challenge page.
- Tenesa A, Haley CS. Nat Rev Genet 2013;14(2):139–149. doi:10.1038/nrg3377.
  Need: the relatives-estimation box and the PO-vs-sib common-environment
  statement.
- Lichtenstein 2009 web appendix (Table 1's $F=1$ for PO in half-sib families);
  Sandin 2014 eAppendix (C coefficients for MHS vs PHS and cousins); Krebs 2024
  Supplementary Notes S1/S2 and Fig. S7.
- `% TODO: verify` Falconer DS 1965 "Maternal effects and selection response",
  Genetics Today vol. 3 pp 763–774 (Pergamon): no DOI; details from the
  reference lists of RRA85 and Kuijper & Hoyle 2015 only.
- `% TODO: verify` Otto, Christiansen & Feldman 1995 Morrison Institute working
  paper: not found; only Feldman, Otto & Christiansen, "Genes, Culture, and
  Inequality", doi:10.1515/9780691190334-006, pp 61–86.
- `% TODO: verify` Fulker DW 1988 chapter (Weir et al. eds, Sinauer): no
  catalogue hit; Fulker 1982 [Fu82]
  not read; Neale & Cardon 1992 / Neale & Maes book chapter not read.
- Maes et al. 2009 (Flexible Mx), Maes et al. 2018 [M18] and 2023 were read from PMC
  HTML, not PDFs; their quotes in the sub-agent findings are UNVERIFIED at the
  page level. Register papers L09, Ke15, Ke16, Ke20, S14, S17 likewise (PMC
  author manuscripts via HTML; no page anchors).

## 7. Open questions

1. Should simACE follow the Washington convention under AM (fresh variance fixed
   at the random-mating value, $V_C$ inflates) instead of Herzig's (total fixed)?
   The ADR chose total-fixed for the reasons in "Additive variance (rejected)";
   this note finds published precedent for both. The choice changes every
   realized correlation under AM + transmission and should be stated as a
   deliberate departure from RCR78/CRR79.
2. The equilibrium $A$–$C$ covariance under recurrent AM in simACE units
   (CRR79 eq 18.1 / eq 19) has not been checked against a simACE run or the toy;
   the one-generation value has. Needed before any validator asserts it.
3. Cloninger III eq 20 (half-sibs under four rearing arrangements) was not
   transcribed; specialising it to "reared together by parents of one" versus
   "reared apart by own parents" would give the published analogue of simACE's
   MHS/PHS pair, if any.
4. RMC79 Table 3's paternal-half-sib rows (the $b^2$ term for half-sibs reared
   apart, and an undefined symbol $\nu$) could not be reconciled with Table 1;
   UNVERIFIED.
5. H26 maps the whole environment $E$ to transmission; simACE transmits only $C$
   with an untransmitted $E$ alongside. Whether H26 supp. eq 93 and eq 31 carry
   over exactly with $e^2 \mapsto C$ and the extra $E$ absorbed into $\sigma^2$
   is UNVERIFIED (plausible by construction, since the untransmitted $E$ is
   independent of everything else, but the $a(t)/a(t+1)$ factor in eq 31 assumes
   $\mathrm{Var}(A)$ is the only changing variance).
6. Does any register design report cousins through sisters versus cousins
   through brothers? None was found; Ke18 reports same-sex versus opposite-sex
   PO and grandparent correlations only.
7. EPIMIGHT has no publication; its environment term can only be cited from the
   repository (`docs/concepts/related-software.md`).

## References (verified 2026-10-14; C = Crossref, P = PubMed esummary)

- [RCR78] Rice J, Cloninger CR, Reich T. Multifactorial inheritance with cultural transmission and assortative mating. I. Description and basic properties of the unitary models. Am J Hum Genet 1978;30(6):618–643. PMID 747189, PMC1685878. (P)
- [CRR79] Cloninger CR, Rice J, Reich T. Multifactorial inheritance with cultural transmission and assortative mating. II. A general model of combined polygenic and cultural inheritance. Am J Hum Genet 1979;31(2):176–198. PMID 453202, PMC1685756. (P)
- [CRR79b] Cloninger CR, Rice J, Reich T. Multifactorial inheritance with cultural transmission and assortative mating. III. Family structure and the analysis of separation experiments. Am J Hum Genet 1979;31(3):366–388. PMID 572636, PMC1685778. (P)
- [RCR80] Rice J, Cloninger CR, Reich T. Analysis of behavioral traits in the presence of cultural transmission and assortative mating: applications to IQ and SES. Behav Genet 1980;10(1):73–92. doi:10.1007/BF01067320. (C; not read)
- [RMC79] Rao DC, Morton NE, Cloninger CR. Path analysis under generalized assortative mating. I. Theory. Genet Res 1979;33(2):175–188. doi:10.1017/S0016672300018310. (C)
- [RMY76] Rao DC, Morton NE, Yee S. Resolution of cultural and biological inheritance by path analysis. Am J Hum Genet 1976;28(3):228–242. PMID 944529. (P)
- [RMY74] Rao DC, Morton NE, Yee S. Analysis of family resemblance. II. A linear model for familial correlation. Am J Hum Genet 1974;26(3):331–359. PMID 4857114. (P)
- [M74] Morton NE. Analysis of family resemblance. I. Introduction. Am J Hum Genet 1974;26(3):318–330. PMID 4827365. (P)
- [CSF73] Cavalli-Sforza LL, Feldman MW. Cultural versus biological inheritance: phenotypic transmission from parents to children. Am J Hum Genet 1973;25(6):618–637. PMID 4797967. (P)
- [CSF78] Cavalli-Sforza LL, Feldman MW. The evolution of continuous variation. III. Joint transmission of genotype, phenotype and environment. Genetics 1978;90(2):391–425. doi:10.1093/genetics/90.2.391. (C; not read)
- [FCS79] Feldman MW, Cavalli-Sforza LL. Aspects of variance and covariance analysis with cultural inheritance. Theor Popul Biol 1979;15(3):276–307. doi:10.1016/0040-5809(79)90043-1. (C; paywalled)
- [ELYM78] Eaves LJ, Last KA, Young PA, Martin NG. Model-fitting approaches to the analysis of human behaviour. Heredity 1978;41(3):249–320. doi:10.1038/hdy.1978.101. (C)
- [K09] Keller MC, Medland SE, Duncan LE, Hatemi PK, Neale MC, Maes HHM, Eaves LJ. Modeling extended twin family data I: description of the Cascade model. Twin Res Hum Genet 2009;12(1):8–18. doi:10.1375/twin.12.1.8. (C)
- [MK09] Medland SE, Keller MC. Modeling extended twin family data II: power associated with different family structures. Twin Res Hum Genet 2009;12(1):19–25. doi:10.1375/twin.12.1.19. (C)
- [K10] Keller MC, Medland SE, Duncan LE. Are extended twin family designs worth the trouble? A comparison of the bias, precision, and accuracy of parameters estimated in four twin family models. Behav Genet 2010;40(3):377–393. doi:10.1007/s10519-009-9320-x. (C)
- [CK05] Coventry WL, Keller MC. Estimating the extent of parameter bias in the classical twin design: a comparison of parameter estimates from extended twin-family and classical twin designs. Twin Res Hum Genet 2005;8(3):214–223. doi:10.1375/twin.8.3.214. (C)
- [KC05] Keller MC, Coventry WL. Quantifying and addressing parameter indeterminacy in the classical twin design. Twin Res Hum Genet 2005;8(3):201–213. doi:10.1375/twin.8.3.201. (C)
- [E99] Eaves L, Heath A, Martin N, Maes H, Neale M, Kendler K, Kirk K, Corey L. Comparing the biological and cultural inheritance of personality and social attitudes in the Virginia 30,000 study of twins and their relatives. Twin Res 1999;2(2):62–80. doi:10.1375/twin.2.2.62. (C)
- [T94] Truett KR, Eaves LJ, Walters EE, Heath AC, Hewitt JK, Meyer JM, Silberg J, Neale MC, Martin NG, Kendler KS. A model system for analysis of family resemblance in extended kinships of twins. Behav Genet 1994;24(1):35–49. doi:10.1007/BF01067927. (C; paywalled)
- [HKEM85] Heath AC, Kendler KS, Eaves LJ, Markell D. The resolution of cultural and biological inheritance: informativeness of different relationships. Behav Genet 1985;15(5):439–465. doi:10.1007/BF01066238. (C; paywalled)
- [M06] Maes HH, Neale MC, Kendler KS, Martin NG, Heath AC, Eaves LJ. Genetic and cultural transmission of smoking initiation: an extended twin kinship model. Behav Genet 2006;36(6):795–808. doi:10.1007/s10519-006-9085-4. (C)
- [M09] Maes HH, Neale MC, Medland SE, Keller MC, Martin NG, Heath AC, Eaves LJ. Flexible Mx specification of various extended twin kinship designs. Twin Res Hum Genet 2009;12(1):26–34. doi:10.1375/twin.12.1.26. (C; HTML only)
- [M23] Maes HH, Neale MC, Ohlsson H, Sundquist J, Sundquist K, Kendler KS. Genetic and cultural transmission of alcohol use disorder in Swedish twin pedigrees. J Stud Alcohol Drugs 2023;84(3):368–377. doi:10.15288/jsad.22-00097. (C; HTML only)
- [H26] Herzig AF, Noûs C, Saint Pierre A, Perdry H. Impact of co-occurrent assortative mating and vertical cultural transmission on measures of genetic associations. Theor Popul Biol 2026;170:26–35. doi:10.1016/j.tpb.2026.06.003. Supplement mmc1.pdf; code github.com/HervePerdry/AMVCTpaper. (C)
- [LK90] Lande R, Kirkpatrick M. Selection response in traits with maternal inheritance. Genet Res 1990;55(3):189–197. doi:10.1017/S0016672300025520. (C)
- [KL89] Kirkpatrick M, Lande R. The evolution of maternal characters. Evolution 1989;43(3):485–503. doi:10.1111/j.1558-5646.1989.tb04247.x. (C; paywalled)
- [RRA85] Riska B, Rutledge JJ, Atchley WR. Covariance between direct and maternal genetic effects in mice, with a model of persistent environmental influences. Genet Res 1985;45(3):287–297. doi:10.1017/S0016672300022278. (C)
- [KT97] Koerhuis ANM, Thompson R. Models to estimate maternal effects for juvenile body weight in broiler chickens. Genet Sel Evol 1997;29(2):225–249. doi:10.1186/1297-9686-29-2-225. (C; pages from the PDF header)
- [V15] Varona L, Munilla S, Casellas J, Moreno C, Altarriba J. Consequences of paternally inherited effects on the genetic evaluation of maternal effects. Genet Sel Evol 2015;47:63. doi:10.1186/s12711-015-0141-5. (C; article number from Europe PMC)
- [W63] Willham RL. The covariance between relatives for characters composed of components contributed by related individuals. Biometrics 1963;19(1):18–27. doi:10.2307/2527570. (C; paywalled)
- [W72] Willham RL. The role of maternal effects in animal breeding: III. Biometrical aspects of maternal effects in animals. J Anim Sci 1972;35(6):1288–1293. doi:10.2527/jas1972.3561288x. (C; blocked)
- [T76] Thompson R. The estimation of maternal genetic variances. Biometrics 1976;32(4):903–917. doi:10.2307/2529273. (C; paywalled; end page from the sub-agent's Crossref query, start page 903 confirmed)
- [Me92] Meyer K. Bias and sampling covariances of estimates of variance components due to maternal effects. Genet Sel Evol 1992;24(6):487–509. doi:10.1186/1297-9686-24-6-487. (C; not retrieved)
- [B14] Bijma P. The quantitative genetics of indirect genetic effects: a selective review of modelling issues. Heredity 2014;112(1):61–69. doi:10.1038/hdy.2013.15. (C)
- [KH15] Kuijper B, Hoyle RB. When to rely on maternal effects and when on phenotypic plasticity? Evolution 2015;69(4):950–968. doi:10.1111/evo.12635. (C)
- [BKK21] Balbona JV, Kim Y, Keller MC. Estimation of parental effects using polygenic scores. Behav Genet 2021;51(3):264–278. doi:10.1007/s10519-020-10032-w. (C)
- [KBK21] Kim Y, Balbona JV, Keller MC. Bias and precision of parameter estimates from models using polygenic scores to estimate environmental and genetic parental influences. Behav Genet 2021;51(3):279–288. doi:10.1007/s10519-020-10033-9. (C)
- [K18] Kong A, Thorleifsson G, Frigge ML, Vilhjalmsson BJ, Young AI, Thorgeirsson TE, Benonisdottir S, Oddsson A, Halldorsson BV, Masson G, Gudbjartsson DF, Helgason A, Bjornsdottir G, Thorsteinsdottir U, Stefansson K. The nature of nurture: effects of parental genotypes. Science 2018;359(6374):424–428. doi:10.1126/science.aan6877. (C)
- [Y18] Young AI, Frigge ML, Gudbjartsson DF, Thorleifsson G, Bjornsdottir G, Sulem P, Masson G, Thorsteinsdottir U, Stefansson K, Kong A. Relatedness disequilibrium regression estimates heritability without environmental bias. Nat Genet 2018;50(9):1304–1310. doi:10.1038/s41588-018-0178-9. (C)
- [E21] Eilertsen EM, Jami ES, McAdams TA, Hannigan LJ, Havdahl AS, Magnus P, Evans DM, Ystrom E. Direct and indirect effects of maternal, paternal, and offspring genotypes: Trio-GCTA. Behav Genet 2021;51(2):154–161. doi:10.1007/s10519-020-10036-6. (C)
- [N24] Nivard MG, Belsky DW, Harden KP, Baier T, Andreassen OA, Ystrøm E, van Bergen E, Lyngstad TH. More than nature and nurture, indirect genetic effects on children's academic achievement are consequences of dynastic social processes. Nat Hum Behav 2024;8(4):771–778. doi:10.1038/s41562-023-01796-2. (C; preprint read)
- [B22] Border R, Athanasiadis G, Buil A, Schork AJ, Cai N, Young AI, Werge T, Flint J, Kendler KS, Sankararaman S, Dahl AW, Zaitlen NA. Cross-trait assortative mating is widespread and inflates genetic correlation estimates. Science 2022;378(6621):754–761. doi:10.1126/science.abo2059. (C)
- [Ke21] Kemper KE, Yengo L, Zheng Z, Abdellaoui A, Keller MC, Goddard ME, Wray NR, Yang J, Visscher PM. Phenotypic covariance across the entire spectrum of relatedness for 86 billion pairs of individuals. Nat Commun 2021;12:1050. doi:10.1038/s41467-021-21283-4. (C; not read)
- [F65] Falconer DS. The inheritance of liability to certain diseases, estimated from the incidence among relatives. Ann Hum Genet 1965;29(1):51–76. doi:10.1111/j.1469-1809.1965.tb00500.x. (C)
- [F67] Falconer DS. The inheritance of liability to diseases with variable age of onset, with particular reference to diabetes mellitus. Ann Hum Genet 1967;31(1):1–20. doi:10.1111/j.1469-1809.1967.tb02015.x. (C)
- [L09] Lichtenstein P, Yip BH, Björk C, Pawitan Y, Cannon TD, Sullivan PF, Hultman CM. Common genetic determinants of schizophrenia and bipolar disorder in Swedish families: a population-based study. Lancet 2009;373(9659):234–239. doi:10.1016/s0140-6736(09)60072-6. (C; HTML only)
- [P19] Pettersson E, Lichtenstein P, Larsson H, Song J, et al. Genetic influences on eight psychiatric disorders based on family data of 4 408 646 full and half-siblings, and genetic data of 333 748 cases and controls. Psychol Med 2019;49(7):1166–1173. doi:10.1017/s0033291718002039. (C; consortium author list truncated)
- [S14] Sandin S, Lichtenstein P, Kuja-Halkola R, Larsson H, Hultman CM, Reichenberg A. The familial risk of autism. JAMA 2014;311(17):1770–1777. doi:10.1001/jama.2014.4144. (C; end page from the sub-agent)
- [S17] Sandin S, Lichtenstein P, Kuja-Halkola R, Hultman C, Larsson H, Reichenberg A. The heritability of autism spectrum disorder. JAMA 2017;318(12):1182–1184. doi:10.1001/jama.2017.12141. (C; end page from the sub-agent)
- [Ke15] Kendler KS, Ohlsson H, Sundquist J, Sundquist K. Triparental families: a new genetic-epidemiological design applied to drug abuse, alcohol use disorders, and criminal behavior in a Swedish national sample. Am J Psychiatry 2015;172(6):553–560. doi:10.1176/appi.ajp.2014.14091127. (C)
- [Ke16] Kendler KS, Ohlsson H, Edwards AC, Lichtenstein P, Sundquist K, Sundquist J. A novel sibling-based design to quantify genetic and shared environmental effects: application to drug abuse, alcohol use disorder and criminal behavior. Psychol Med 2016;46(8):1639–1650. doi:10.1017/s003329171500224x. (C)
- [Ke18] Kendler KS, Ohlsson H, Sundquist J, Sundquist K. Transmission of alcohol use disorder across three generations: a Swedish National Study. Psychol Med 2018;48(1):33–42. doi:10.1017/s0033291717000794. (C)
- [Ke20] Kendler KS, Ohlsson H, Sundquist J, Sundquist K. Maternal half-sibling families with discordant fathers: a contrastive design assessing cross-generational paternal genetic transmission of alcohol use disorder, drug abuse and major depression. Psychol Med 2020;50(6):973–980. doi:10.1017/s0033291719000874. (C)
- [WG12] Wray NR, Gottesman II. Using summary data from the Danish National Registers to estimate heritabilities for schizophrenia, bipolar disorder, and major depressive disorder. Front Genet 2012;3:118. doi:10.3389/fgene.2012.00118. (C lists Wray only; second author from the PDF)
- [A22] Athanasiadis G, Meijsen JJ, Helenius D, Schork AJ, Ingason A, Thompson WK, Geschwind DH, Werge T, Buil A. A comprehensive map of genetic relationships among diagnostic categories based on 48.6 million relative pairs from the Danish genealogy. Proc Natl Acad Sci USA 2022;119(6):e2118688119. doi:10.1073/pnas.2118688119. (C)
- [Kr24] Dybdahl Krebs M, Georgii Hellberg K-L, Lundberg M, et al. Genetic liability estimated from large-scale family data improves genetic prediction, risk score profiling, and gene mapping for major depression. Am J Hum Genet 2024;111(11):2494–2509. doi:10.1016/j.ajhg.2024.09.009. (C; author list truncated)
- [Z13] Zaitlen N, Kraft P, Patterson N, Pasaniuc B, Bhatia G, Pollack S, Price AL. Using extended genealogy to estimate components of heritability for 23 quantitative and dichotomous traits. PLoS Genet 2013;9(5):e1003520. doi:10.1371/journal.pgen.1003520. (C)
- [X16] Xia C, Amador C, Huffman J, Trochet H, Campbell A, Porteous D, Generation Scotland, Hastie ND, Hayward C, Vitart V, Navarro P, Haley CS. Pedigree- and SNP-associated genetics and recent environment are the major contributors to anthropometric and cardiometabolic trait variation. PLoS Genet 2016;12(2):e1005804. doi:10.1371/journal.pgen.1005804. (C)
- [TH13] Tenesa A, Haley CS. The heritability of human disease: estimation, uses and abuses. Nat Rev Genet 2013;14(2):139–149. doi:10.1038/nrg3377. (C; paywalled)
- [MG14] McGlothlin JW, Galloway LF. The contribution of maternal effects to selection response: an empirical test of competing models. Evolution 2014;68(2):549–558. doi:10.1111/evo.12235. (C; main text paywalled, author-hosted supplement read)
- [R96] Robinson DL. Models which might explain negative correlations between direct and maternal genetic effects. Livest Prod Sci 1996;45(2–3):111–122. doi:10.1016/0301-6226(96)00002-4. (C; not read; KT97 cites the 1994 Guelph proceedings version)
- [M18] Maes HH, Morley K, Neale MC, Kendler KS, Heath AC, Eaves LJ, Martin NG. Cross-cultural comparison of genetic and cultural transmission of smoking initiation using an extended twin kinship model. Twin Res Hum Genet 2018;21(3):179–190. doi:10.1017/thg.2018.22. (C; HTML only)
- [Fu82] Fulker DW. Extensions of the classical twin method. Prog Clin Biol Res 1982;103 Pt A:395–406. PMID 6891786. (P; not read)

Per-paper sub-agent findings (`findings_{path_models,twin_family,maternal,nurture,registry}.md`) are session-local and not in the repo; everything this note relies on from them is restated above with its page or equation number.
