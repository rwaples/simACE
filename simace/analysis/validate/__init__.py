"""On-demand checks: what ``simace validate`` runs on finished reps and ``simace run`` never does.

Every calibrated test and its verdict across reps lives here, plus any
quality check that needs outcomes a rep does not store (ADR 0028). Analyze's
quality checks are in :mod:`simace.analysis.quality`.

- :mod:`.calibrated` — ``RepInputs``, ``CalibratedTest`` and the per-cell results
- :mod:`.on_demand` — the check registry, ``run_on_demand`` and the ``validation.yaml`` schema
- :mod:`.onset_tau` — onset τ for frailty traits (ADR 0027)
- :mod:`.runner` — the per-rep ``simace validate --checks`` primitive
- :mod:`.schema_file` — reading ``validation.yaml`` and ``validation_summary.yaml``, refusing files without their schema
- :mod:`.verdict` — each test's rule across reps, the Bonferroni families, ``validation_summary.yaml``
"""
