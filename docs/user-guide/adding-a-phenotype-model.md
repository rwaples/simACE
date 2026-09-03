# Adding a phenotype model

Each model family is a frozen dataclass under `simace/phenotype/models/` that
subclasses `PhenotypeModel`. The
`MODELS` dict in `simace/phenotype/models/__init__.py` is the registry.
The dispatcher, the config validator, and the CLI all read from it.

1. Create `simace/phenotype/models/my_model.py` with a class
   `MyModel(PhenotypeModel)`.
2. Give the class typed parameter fields.
3. Implement the abstract methods `from_config`, `add_cli_args`, `from_cli`,
   `cli_flag_attrs`, `to_params_dict`, and `simulate`.
4. Validate the parameters in `__post_init__`.
5. In `from_config` and `from_cli`, wrap the construction in
   `wrap_trait_error` from `simace/phenotype/models/_base.py`. A `ValueError`
   or `TypeError` then names the trait.
6. Import the class in `simace/phenotype/models/__init__.py`.
7. Add `"my_model": MyModel` to `MODELS`.

After step 7, `_simulate_one_trait` in `simace/phenotype/runner.py`,
`_validate_phenotype_config` in `simace/config.py`, and the `simace-phenotype`
CLI accept the new `model` value without further changes.

Add the model to the tables in [Phenotype models](phenotype-models.md).
