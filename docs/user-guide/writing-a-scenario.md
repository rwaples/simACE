# Writing a scenario

Add a top-level entry to `config/{folder}.yaml`. `simace run` discovers these
files automatically, except files whose names start with `_`. The file name
sets the output folder unless a scenario sets `folder` explicitly. A new file
creates a new folder by default.

Set only the values that differ from `config/_default.yaml`. Sections merge
with the defaults field by field. For example, `high_heritability` in
`config/heritability.yaml` overrides the seed and both traits' variance
components:

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

This scenario writes under `results/heritability/high_heritability/`. It
inherits every default that the YAML entry omits. [Configuration](configuration.md)
lists all parameters and defaults. [Running the pipeline](running-the-pipeline.md)
shows how to target a scenario.
