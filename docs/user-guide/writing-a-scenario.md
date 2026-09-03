# Writing a scenario

To add a scenario, add one dictionary to the scenario file for its folder,
`config/{folder}.yaml`. A new folder needs a new file. This is the start of
`config/base.yaml`:

```yaml
baseline10K:
  seed: 1042
  N: 10000
baseline100K:
  seed: 2042
  N: 100000
baseline100K_sample5K:
  seed: 2042
  N: 100000
  ascertainment:
    N_sample: 5000
```

Set only the values that differ from `config/_default.yaml`. Sections merge
over the defaults field by field, so a scenario can change one value inside a
section and inherit the rest. `high_heritability` in
`config/heritability.yaml` changes only the variance components:

```yaml
high_heritability:
  seed: 4042
  pedigree:
    trait1:
      A: 0.8
      C: 0.0
      E: 0.2
    trait2:
      A: 0.8
      C: 0.0
      E: 0.2
```

[Configuration](configuration.md) lists every parameter with its default. To
run the scenario, target its folder and name as described in
[Running the pipeline](running-the-pipeline.md).
