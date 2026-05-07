"""Tests for ``scripts/migrate_prevalence_keys.py``.

Exercises the in-memory ``_walk`` and ``_migrate_trait_block`` helpers on
synthetic YAML structures, plus an end-to-end file rewrite + idempotency
check via ``migrate_file``.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "migrate_prevalence_keys.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("migrate_prevalence_keys", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_prevalence_keys"] = module
    spec.loader.exec_module(module)
    return module


migrate = _load_module()


def test_adult_prevalence_moves_into_params():
    block = {"model": "adult", "params": {"method": "ltm"}, "prevalence": 0.1, "beta": 1.0}
    changed = migrate._migrate_trait_block(block)
    assert changed
    assert "prevalence" not in block
    assert block["params"] == {"method": "ltm", "prevalence": 0.1}


def test_cure_frailty_prevalence_moves_into_params():
    block = {
        "model": "cure_frailty",
        "params": {"distribution": "gompertz", "rate": 0.01},
        "prevalence": {"female": 0.04, "male": 0.08},
    }
    changed = migrate._migrate_trait_block(block)
    assert changed
    assert "prevalence" not in block
    assert block["params"]["prevalence"] == {"female": 0.04, "male": 0.08}


def test_frailty_prevalence_deleted():
    block = {"model": "frailty", "params": {"distribution": "weibull"}, "prevalence": 0.2}
    changed = migrate._migrate_trait_block(block)
    assert changed
    assert "prevalence" not in block
    assert block["params"] == {"distribution": "weibull"}


def test_first_passage_prevalence_deleted():
    block = {"model": "first_passage", "params": {"drift": -0.5, "shape": 1.0}, "prevalence": 0.1}
    changed = migrate._migrate_trait_block(block)
    assert changed
    assert "prevalence" not in block
    assert "prevalence" not in block["params"]


def test_unset_model_defaults_to_frailty():
    """If `model` is omitted, the default is frailty → prevalence is deleted."""
    block = {"params": {"distribution": "weibull"}, "prevalence": 0.1}
    changed = migrate._migrate_trait_block(block)
    assert changed
    assert "prevalence" not in block


def test_already_migrated_block_is_idempotent():
    block = {"model": "adult", "params": {"method": "ltm", "prevalence": 0.1}}
    changed = migrate._migrate_trait_block(block)
    assert not changed
    assert block == {"model": "adult", "params": {"method": "ltm", "prevalence": 0.1}}


def test_walk_handles_default_yaml_shape():
    """`_default.yaml` style: defaults wrapper around phenotype.{trait1,trait2}."""
    data = {
        "defaults": {
            "phenotype": {
                "trait1": {"model": "frailty", "prevalence": 0.1},
                "trait2": {"model": "adult", "params": {"method": "ltm"}, "prevalence": 0.2},
            }
        }
    }
    changed = migrate._walk(data)
    assert changed
    assert "prevalence" not in data["defaults"]["phenotype"]["trait1"]
    assert data["defaults"]["phenotype"]["trait2"]["params"]["prevalence"] == 0.2


def test_walk_handles_per_scenario_yaml_shape():
    """Per-scenario shape: top-level scenario name → phenotype block."""
    data = {
        "demo_a": {
            "phenotype": {
                "trait1": {"model": "cure_frailty", "params": {"distribution": "gompertz"}, "prevalence": 0.05},
            }
        },
        "demo_b": {
            "phenotype": {
                "trait1": {"model": "frailty", "prevalence": 0.1},
            }
        },
    }
    changed = migrate._walk(data)
    assert changed
    assert data["demo_a"]["phenotype"]["trait1"]["params"]["prevalence"] == 0.05
    assert "prevalence" not in data["demo_b"]["phenotype"]["trait1"]


def test_migrate_file_idempotent(tmp_path):
    yaml_text = """\
demo:
  phenotype:
    trait1:
      model: adult
      prevalence: 0.1
      params:
        method: ltm
    trait2:
      model: frailty
      prevalence: 0.2
      params:
        distribution: weibull
        scale: 100
        rho: 2.0
"""
    path = tmp_path / "demo.yaml"
    path.write_text(yaml_text)

    assert migrate.migrate_file(path) is True
    assert migrate.migrate_file(path) is False  # idempotent

    import yaml

    out = yaml.safe_load(path.read_text())
    t1 = out["demo"]["phenotype"]["trait1"]
    t2 = out["demo"]["phenotype"]["trait2"]
    assert "prevalence" not in t1
    assert t1["params"]["prevalence"] == 0.1
    assert "prevalence" not in t2
    assert "prevalence" not in t2["params"]


def test_repo_configs_are_migrated():
    """The repo's own config/*.yaml should already be in the post-PR3 shape."""
    config_dir = REPO_ROOT / "config"
    assert config_dir.exists()
    pending = []
    for path in sorted(config_dir.glob("*.yaml")):
        if path.name in migrate.SKIP_FILES:
            continue
        import copy

        import yaml

        data = yaml.safe_load(path.read_text())
        if data is None:
            continue
        probe = copy.deepcopy(data)
        if migrate._walk(probe):
            pending.append(path.name)
    assert pending == [], f"these configs are not yet migrated: {pending}"
