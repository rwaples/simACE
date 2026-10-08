"""Phenotype model registry.

To add a new phenotype model:

  1. Write a new module under ``simace/phenotype/models/`` defining a
     class that subclasses ``PhenotypeModel`` (see ``_base.py``).
  2. Implement the abstract methods (``_from_config``, ``add_cli_args``,
     ``from_cli``, ``cli_flag_attrs``, ``to_params_dict``, ``simulate``).
     Do not override ``from_config``: the inherited one rejects unknown
     ``phenotype_params`` keys and adds trait context before ``_from_config``.
  3. Add the model's name to ``MODEL_FAMILIES`` and its accepted keys to
     ``_MODEL_KEYS`` in ``simace/core/phenotype_keys.py``.
  4. Import the class here and add it to the ``MODELS`` dict below.

That's the entire surface — there is no auto-discovery and no decorator.
The hand-authored dict is the single source of truth for which model names
the dispatcher accepts.
"""

from simace.phenotype.models._base import PhenotypeModel
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
]


MODELS: dict[str, type[PhenotypeModel]] = {
    cls.name: cls for cls in (FrailtyModel, CureFrailtyModel, AdultModel, FirstPassageModel, SimpleLtmModel)
}
