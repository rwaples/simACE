# Related software

simACE exists because no simulator we could find produces all four of
registry-scale pedigrees, ACE liabilities, age-at-onset with censoring, and
ascertainment in one run. This page is the evidence for that claim and the
source list for the comparison section of the application note.

Each group below solves part of the problem. Every table gives the citation,
what the tool does, and what it does not do that simACE does. For the
variance components themselves, see [ACE model](ace-model.md). For the
estimators simACE is built to benchmark, see [Methods](methods.md).

Every field in every citation comes from Crossref, checked on 2026-09-03.
See [Re-checking the citations](#re-checking-the-citations) for the command.

## Disease pedigrees with age-at-onset and ascertainment

These two tools come nearest. Both put a liability model, an onset age, and
family structure into one simulation, the combination simACE needs. Neither
one scales to a registry.

| Tool | Citation | What it does | Gap relative to simACE |
|---|---|---|---|
| SimRVPedigree | Nieuwoudt, Jones, Brooks-Wilson, Graham (2018). *Source Code Biol Med* 13:2. [doi:10.1186/s13029-018-0069-6](https://doi.org/10.1186/s13029-018-0069-6) | Simulates families ascertained for multiple affected relatives, with age-at-onset and right-censoring. | Rare-variant model, one family at a time, no common environment (C). |
| Multifactorial Disease Risk Calculator | Campbell, Li, Sham (2018). *Genet Epidemiol* 42(2):130–133. [doi:10.1002/gepi.22101](https://doi.org/10.1002/gepi.22101) | Applies a liability threshold with an ACE variance partition and age-at-onset curves. | Computes risk for a pedigree you supply. Does not simulate pedigrees. |

## Pedigree-structure and gene-drop simulators

All four tools below produce pedigree structures, genotypes, segments shared
identical by descent (IBD), or some combination of those. None of them
simulates a phenotype, so the gap relative to simACE is the same for every
row: no liability model, no onset, no censoring, no ascertainment.

| Tool | Citation | What it does |
|---|---|---|
| Ped-sim | Caballero et al. (2019). *PLOS Genet* 15(12):e1007979. [doi:10.1371/journal.pgen.1007979](https://doi.org/10.1371/journal.pgen.1007979) | Drops IBD segments through pedigree structures you specify, using sex-specific genetic maps. |
| py_ped_sim | Guardado et al. (2025). *BMC Bioinformatics* 26:122. [doi:10.1186/s12859-025-06142-z](https://doi.org/10.1186/s12859-025-06142-z) | Simulates pedigree structures forward in time, then genomes through SLiM. |
| SLINK | Schäffer, Lemire, Ott, Lathrop, Weeks (2011). *Hum Hered* 71(2):126–134. [doi:10.1159/000324177](https://doi.org/10.1159/000324177) | Simulates markers conditional on an observed trait, for linkage analysis. |
| ibdsim2, part of the pedsuite | Vigeland (2021). *Pedigree Analysis in R*. ISBN 9780128244302. | Simulates IBD segments and provides pedigree utilities in R. |

## Forward-time and breeding simulators with quantitative phenotypes

This group gets closest on scale and on the additive genetic component.
The two agricultural packages, AlphaSimR and pedSimulate, already carry
breeding values through many generations. What they lack is the binary,
censored, age-dependent outcome that a health registry actually records.

| Tool | Citation | What it does | Gap relative to simACE |
|---|---|---|---|
| AlphaSimR | Gaynor, Gorjanc, Hickey (2021). *G3* 11(2):jkaa017. [doi:10.1093/g3journal/jkaa017](https://doi.org/10.1093/g3journal/jkaa017) | Simulates breeding programs with additive and non-additive traits. | No common environment (C), no liability threshold, no onset. |
| pedSimulate | Nilforooshan (2022). *Rev Bras Zootec* 51:e20210131. [doi:10.37496/rbz5120210131](https://doi.org/10.37496/rbz5120210131) | Simulates pedigrees, breeding values, phenotypes, and assortative mating. | Written for animal breeding. No onset, no censoring. |
| simuPOP | Peng, Kimmel (2005). *Bioinformatics* 21(18):3686–3687. [doi:10.1093/bioinformatics/bti584](https://doi.org/10.1093/bioinformatics/bti584) | Simulates populations forward in time, scripted from Python. | Locus-based. No variance-component phenotypes. |
| simuPOP applied to complex disease | Peng, Amos, Kimmel (2007). *PLoS Genet* 3(3):e47. [doi:10.1371/journal.pgen.0030047](https://doi.org/10.1371/journal.pgen.0030047) | Simulates a population, then draws pedigrees and onset ages from that population. | Locus-based disease model. |
| SLiM 4 | Haller, Messer (2023). *Am Nat* 201(5):E127–E139. [doi:10.1086/723601](https://doi.org/10.1086/723601) | Simulates whole genomes forward in time. | Runs on a fixed pedigree, but has no liability-threshold onset model. |
| msprime 1.0 | Baumdicker et al. (2022). *Genetics* 220(3):iyab229. [doi:10.1093/genetics/iyab229](https://doi.org/10.1093/genetics/iyab229) | Simulates ancestry under the coalescent and under Wright-Fisher, including on a fixed pedigree. | Genomes only. No phenotype. |

The 2007 simuPOP paper is the closest thing to a precedent for the simACE
design. It simulates a whole population first, then samples pedigrees and
onset ages out of it. The disease model is locus-based rather than a
variance partition, so the ground truth it can supply is not the parameter
an ACE estimator reports.

## Twin-data ACE simulators

Twin packages model A, C, and E directly, and two of the three simulate
data. The limit is the study design. A twin pair or a nuclear family is not
a genealogy that spans several generations.

| Tool | Citation | What it does | Gap relative to simACE |
|---|---|---|---|
| umx | Bates, Maes, Neale (2019). *Twin Res Hum Genet* 22(1):27–41. [doi:10.1017/thg.2019.2](https://doi.org/10.1017/thg.2019.2) | Generates twin data with A, C, and E under your control. | Twin pairs only. No onset, no censoring. |
| mets | Holst, Scheike, Hjelmborg (2016). *Comput Stat Data Anal* 93:324–335. [doi:10.1016/j.csda.2015.01.014](https://doi.org/10.1016/j.csda.2015.01.014) | Simulates twin data and fits the censored liability threshold model. | Twin pairs only. No extended pedigrees. |
| OpenMx 2.0 | Neale et al. (2016). *Psychometrika* 81(2):535–549. [doi:10.1007/s11336-014-9435-8](https://doi.org/10.1007/s11336-014-9435-8) | Fits structural equation models, including twin models. | Fits models. Does not simulate pedigrees. |

## Registry-scale methods that need a testbed

These are the estimators simACE is built to test, not competitors to it.
Each paper validates its method on a simulation written for that paper
alone. A simulator that several methods agree to be judged against is what
the application note argues for.

| Method | Citation |
|---|---|
| LT-FH | Hujoel, Gazal, Loh, Patterson, Price (2020). *Nat Genet* 52(5):541–547. [doi:10.1038/s41588-020-0613-6](https://doi.org/10.1038/s41588-020-0613-6) |
| LT-FH++ | Pedersen et al. (2022). *Am J Hum Genet* 109(3):417–432. [doi:10.1016/j.ajhg.2022.01.009](https://doi.org/10.1016/j.ajhg.2022.01.009) |
| PA-FGRS | Krebs et al. (2024). *Am J Hum Genet* 111(11):2494–2509. [doi:10.1016/j.ajhg.2024.09.009](https://doi.org/10.1016/j.ajhg.2024.09.009) |
| TetraHer | Speed, Evans (2024). *Am J Hum Genet* 111(4):680–690. [doi:10.1016/j.ajhg.2024.02.010](https://doi.org/10.1016/j.ajhg.2024.02.010) |
| Danish genealogy relative-pair analysis | Athanasiadis et al. (2022). *Proc Natl Acad Sci* 119(6):e2118688119. [doi:10.1073/pnas.2118688119](https://doi.org/10.1073/pnas.2118688119) |
| Nationwide heritability map | Auning et al. (2026). *Nat Commun* 17:4080. [doi:10.1038/s41467-026-69991-z](https://doi.org/10.1038/s41467-026-69991-z) |

## What still needs care before you cite it

EPIMIGHT has no citation. The
[BioPsyk/epimight](https://github.com/BioPsyk/epimight) README listed no
paper, no preprint, and no DOI on 2026-09-03. Cite the repository URL until
a publication appears.

Two entries carry a year that depends on which version you mean. Crossref
dates AlphaSimR to 2020 online and to 2021 in print. It dates Campbell and
colleagues to 2017 online and to 2018 in print. This page and the BibTeX
file both use the print year.

Author lists are partly checked. Where a row reads "et al." here, or an
entry reads "and others" in the BibTeX file, the check covered the first
three authors and stopped.

## Re-checking the citations

`docs/concepts/related-software.bib` holds one BibTeX entry per row above.
To confirm every DOI still resolves at Crossref, run:

```bash
grep -o 'doi *= *{[^}]*}' docs/concepts/related-software.bib \
	| sed 's/.*{\(.*\)}/\1/' \
	| while read -r d; do
		printf '%s  %s\n' \
			"$(curl -s -o /dev/null -w '%{http_code}' "https://api.crossref.org/works/$d")" "$d"
	done
```

Every line should start with `200`. The file holds 21 entries and 20 DOIs,
because the Vigeland book carries an ISBN instead. Regenerate those two
counts with `grep -c '^@' docs/concepts/related-software.bib` and
`grep -c 'doi *= *{' docs/concepts/related-software.bib`.
