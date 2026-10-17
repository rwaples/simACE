"""On-demand checks: what ``simace validate`` runs on finished reps and ``simace run`` never does.

Every calibrated test lives here, plus any quality check that needs outcomes a
rep does not store (ADR 0028). Analyze's quality checks are in
:mod:`simace.analysis.quality`.

- :mod:`.on_demand` — the check registry and ``run_on_demand``
- :mod:`.onset_tau` — onset τ for frailty traits (ADR 0027)
- :mod:`.runner` — the per-rep ``simace validate --checks`` primitive
"""
