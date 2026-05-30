"""Phenotype simulation for two correlated traits.

Each trait is simulated by one of the four model families registered in
``simace.phenotype.models``:

  * ``frailty``       — proportional hazards frailty (baseline hazards live
                        in ``simace.phenotype.hazards``).
  * ``cure_frailty``  — mixture cure model: threshold determines case
                        status, frailty determines onset time among cases.
  * ``adult``         — ADuLT age-dependent liability threshold.
  * ``first_passage`` — inverse-Gaussian first-passage time.

Adding a new model is a single new file under
``simace/phenotype/models/`` plus one entry in
``simace/phenotype/models/__init__.py``'s ``MODELS`` dict.

The dispatcher (:func:`run_phenotype`) and CLI live in
:mod:`simace.phenotype.runner`; the names below are re-exported for the
public API and the ``simace-phenotype`` entry point.
"""

from .runner import cli, run_phenotype

__all__ = ["cli", "run_phenotype"]
