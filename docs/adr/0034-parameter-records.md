# ADR 0034: Scenario parameters are parsed once per boundary into typed records

## Status

Accepted and implemented 2026-11-15 for issue #86. The plan, its Codex review and the decisions are in
`plans/simace-issue-86.md`.

## Context

- Scenario parameters travelled as a flat `dict[str, Any]`, and each consumer re-parsed the keys it needed with its
  own default. The readings disagreed. #93: the quality checks read `assort1`/`assort2` while the simulator mated
  on the `assort_matrix` diagonal. #94: a null value became the stage CLI's argparse default. #96: phenotype
  parameters passed config load and failed later. Those three were fixed one at a time; this ADR removes the
  structure that let them happen.
- Parameters enter a consumer through five shapes, not one: config load, a rep's `params.yaml`, `report.yaml`'s
  `inputs.parameters`, a gathered TSV row, and fitACE's Snakemake projection. The last two are partial by
  design, so no single complete record can be required at every boundary.
- `params.yaml` is a fitACE contract: its keys may be added, never renamed or removed. Older files lack the keys
  added since: `mating_model` (ADR 0002), `Ctm_*` (ADR 0025) and `inbreeding_*` (ADR 0033). Code changes never
  make a rep stale, so plots still read them.

## Decision

1. **Four records own the parameters' reading.** These live in `simace.simulation.params`, which imports no
   numba:
   - `Span`: `G_ped`, `G_sim` and the burn-in.
   - `VarianceComponents`: A, C and E per trait, and `rA`, `rC`, `rE`.
   - `Assortment`: the mating model and the per-iteration schedule the simulator mates on.
   - `CTransmission`: the C transmission coefficients.

   Each validates its own invariants when built. Each `from_mapping` reads its flat keys and gives every key a
   default only through `LEGACY_DEFAULTS`, the defaults of the keys an older `params.yaml` lacks.
2. **`Assortment` is the only assortative-mating predicate.** `active(t)` and `any_active` read the schedule:
   - Wright-Fisher mates at random.
   - A set `assort_matrix` mates on its diagonal.
   - A per-generation map is resolved over `G_sim`.

   So a zero-diagonal matrix, or a map entry past `G_sim`, is random mating. The closed-form reference-line gate
   follows this. Under C transmission those two cases now get the analytic line, not the recorded truth.
3. **`SimulationParams` is the simulator's input and the parsed `params.yaml`.** It is composed of the four
   records plus `seed`, `N`, `mating_lambda`, `p_mztwin` and the resolved inbreeding plan.
   - `run_simulation` keeps its keyword API for callers outside the stage, builds the record, and calls
     `simulate(params)`, which reads only the record. The simulate stage builds the record from its flags.
   - The analysis parses `params.yaml`, and `simace validate` and the plots parse `report.yaml`, each with
     `SimulationParams.from_mapping`.
   - `from_mapping` checks the records' own invariants but not whether a simulation can run on the
     parameters (`mating_lambda`, `p_mztwin`, C transmission feasibility). Those checks are `create`'s, which
     the simulator and config load run. So a report scored with a deliberately wrong coefficient, as
     simACE-validate's power study does, still parses.
4. **Each boundary parses once and passes records inward.** The quality checks, the recorded-pedigree truth, the
   calibrated tests, the effective-size expectations and the plots take the records. Examples:
   - `reference_corr(type, assortment, c_transmission, vc, t)`;
   - `recorded_matings(span, vc, assortment)`;
   - `am_relatedness_mode(assortment, t)`.

   A partial boundary builds only the records its consumer needs. For the gathered TSV and fitACE's projection
   that is the assortment, C transmission and variance records, and no line is drawn when a column is missing.
5. **Config load builds a `Scenario`.** `simace.config.load_scenarios` returns a `Scenario` per name. Each holds:
   - the scenario's own keys (`explicit`, which the Wright-Fisher override check reads);
   - the flat parameters merged once over the defaults and `LEGACY_DEFAULTS` (`flat`, which the stage command
     lines and `params.yaml` read);
   - the four records.

   `Scenario.parse` runs every config-load check of the scenario, so a scenario outside the records' ranges
   fails at load. The inbreeding cost refusal stays in the simulate stage (ADR 0033 §4).
   `resolve_defaults`, `resolve_scenarios`, `flatten_hierarchical`, `KNOWN_SIM_KEYS` and the `get_*` accessors
   keep their dict API for fitACE.
6. **Dicts remain only at the edges.** These are:
   - on-disk YAML;
   - the boundary parsers;
   - the stage argv builder;
   - the writers of `params.yaml` and the gathered TSV;
   - the atlas title page, which prints configured values.

   `tests/test_param_reads.py` fails on a raw read of an assortment, mating-model, `G_sim`, C transmission or
   inbreeding key outside those modules, the one thing `ty` cannot see.

## Consequences

- One default per key and one assortative-mating predicate. A new consumer that needs a derived fact takes the
  record; the guard test stops it reading the key raw.
- Analysis signatures changed (release 0.4.0). fitACE moved with them. The pinned companions (pg-phenotype's
  `tools/`, simACE-validate) need the new signatures when their pins move; an issue in each lists them.
- Test fixtures build records with `tests.conftest.sim_params`, which fills the keys a test leaves out from
  `config/_default.yaml` under `run_simulation`'s own defaults.
- The `params.yaml` and `emit_params` output of every shipped scenario is unchanged. So is every analysis
  decision taken from it: `tools/scenario_fingerprint.py` compared all 492 scenarios and a 21-scenario pipeline
  subset before and after.
