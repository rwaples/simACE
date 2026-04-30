"""Phenotype model registry.

To add a new phenotype model:

  1. Write a new module under ``simace/phenotyping/models/`` defining a
     class that subclasses ``PhenotypeModel`` (see ``_base.py``).
  2. Implement the abstract methods (``from_config``, ``add_cli_args``,
     ``from_cli``, ``cli_flag_attrs``, ``to_params_dict``, ``simulate``).
  3. Import the class here and add it to the ``MODELS`` dict below.

That's the entire surface — there is no auto-discovery and no decorator.
The hand-authored dict is the single source of truth for which model names
the dispatcher accepts.
"""

from simace.phenotyping.models._base import PhenotypeModel
from simace.phenotyping.models.adult import AdultModel
from simace.phenotyping.models.cure_frailty import CureFrailtyModel
from simace.phenotyping.models.first_passage import FirstPassageModel
from simace.phenotyping.models.frailty import FrailtyModel

__all__ = [
    "MODELS",
    "AdultModel",
    "CureFrailtyModel",
    "FirstPassageModel",
    "FrailtyModel",
    "PhenotypeModel",
]


MODELS: dict[str, type[PhenotypeModel]] = {
    "frailty": FrailtyModel,
    "cure_frailty": CureFrailtyModel,
    "adult": AdultModel,
    "first_passage": FirstPassageModel,
}
