"""Phenotype model registry.

To add a new phenotype model:

  1. Write a new module under ``simace/phenotype/models/`` defining a
     class that subclasses ``PhenotypeModel`` (see ``_base.py``).
  2. Set ``name`` and ``config_keys`` and implement the abstract methods
     (``_from_config``, ``simulate``). Do not override ``from_config``: the
     inherited one rejects keys outside ``config_keys`` and adds trait context
     before ``_from_config``. Validate every value in ``__post_init__``;
     config load calls ``from_config`` for every scenario.
  3. Keep numba out of the module's imports: kernels live in
     ``simace/phenotype/_kernels.py`` and are imported inside ``simulate``.
  4. Import the class here and add it to the ``MODELS`` dict below.

That's the entire surface — there is no auto-discovery and no decorator.
The hand-authored dict is the single source of truth for which model names
the dispatcher accepts.
"""

from simace.phenotype.models._base import PhenotypeModel
from simace.phenotype.models._prevalence import case_status_from_liability
from simace.phenotype.models.adult import AdultModel
from simace.phenotype.models.cure_frailty import CureFrailtyModel
from simace.phenotype.models.first_passage import FirstPassageModel
from simace.phenotype.models.frailty import FrailtyModel
from simace.phenotype.models.simple_ltm import SimpleLtmModel

__all__ = [
    "MODELS",
    "AdultModel",
    "CureFrailtyModel",
    "FirstPassageModel",
    "FrailtyModel",
    "PhenotypeModel",
    "SimpleLtmModel",
    "case_status_from_liability",
]


MODELS: dict[str, type[PhenotypeModel]] = {
    cls.name: cls for cls in (FrailtyModel, CureFrailtyModel, AdultModel, FirstPassageModel, SimpleLtmModel)
}
