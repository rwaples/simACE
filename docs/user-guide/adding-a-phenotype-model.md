# Adding a phenotype model

Each model family is a frozen dataclass under `simace/phenotype/models/` that
subclasses `PhenotypeModel`. The
`MODELS` dict in `simace/phenotype/models/__init__.py` is the registry.
The dispatcher, the config validator, and the CLI all read from it.

1. Create `simace/phenotype/models/my_model.py` with a class
   `MyModel(PhenotypeModel)`.
2. Give the class typed parameter fields, a `name`, and `config_keys`, the
   `phenotype_params` keys it accepts. The inherited `from_config` rejects any
   other key.
3. Implement the abstract methods `_from_config`, `add_cli_args`, `from_cli`,
   `cli_flag_attrs`, `to_params_dict`, and `simulate`.
4. Validate every parameter value in `__post_init__`. Config load builds each
   scenario's models through `from_config`, so a value `__post_init__` accepts
   reaches a job.
5. In `from_cli`, wrap the construction in `wrap_trait_error` from
   `simace/phenotype/models/_base.py`. A `ValueError` or `TypeError` then names
   the trait. `from_config` already does this.
6. Do not import numba at the top of the module. Put kernels in
   `simace/phenotype/_kernels.py` and import them inside `simulate`, so that
   config load and `simace ls` stay fast.
7. Import the class in `simace/phenotype/models/__init__.py`.
8. Add `"my_model": MyModel` to `MODELS`.

After step 8, `_simulate_one_trait` in `simace/phenotype/runner.py`,
`_validate_phenotype_config` in `simace/config.py`, and the `simace phenotype`
and `simace cohort` commands accept the new `model` value without further
changes.

Add the model to the tables in [Phenotype models](phenotype-models.md).
