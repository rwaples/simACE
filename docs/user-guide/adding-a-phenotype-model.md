# Adding a phenotype model

Each model family is a frozen dataclass under `simace/phenotype/models/` that
subclasses `PhenotypeModel`. There is no decorator and no auto-discovery. The
`MODELS` dict in `simace/phenotype/models/__init__.py` is the registry, and
the dispatcher, the config validator, and the CLI all read from it.

1. Create `simace/phenotype/models/my_model.py` with a class
   `MyModel(PhenotypeModel)`. Give it typed parameter fields and implement
   the abstract methods `from_config`, `add_cli_args`, `from_cli`,
   `cli_flag_attrs`, `to_params_dict`, and `simulate`.
2. Validate the parameters in `__post_init__`. In `from_config` and
   `from_cli`, wrap the construction in `wrap_trait_error` from
   `simace/phenotype/models/_base.py` so that a `ValueError` or `TypeError`
   names the trait.
3. Import the class in `simace/phenotype/models/__init__.py` and add
   `"my_model": MyModel` to `MODELS`.

After step 3, `_simulate_one_trait` in `simace/phenotype/runner.py`,
`_validate_phenotype_config` in `simace/config.py`, and the `simace-phenotype`
CLI accept the new `model` value without further changes.

Add the model to the tables in [Phenotype models](phenotype-models.md).
