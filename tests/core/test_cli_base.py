"""Argparse ``type=`` parsers in ``simace.core.cli_base``."""

import argparse

import pytest

from simace.core.cli_base import float_or_generation_map, generation_map


def test_generation_map_keys_are_ints():
    assert generation_map('{"0": 80, "4": 70}') == {0: 80, 4: 70}


def test_float_or_generation_map_takes_a_scalar_or_a_map():
    assert float_or_generation_map("0.5") == 0.5
    assert float_or_generation_map('{"0": 0.5}') == {0: 0.5}


@pytest.mark.parametrize("value", ["[80, 80]", "true", "null", '"x"'])
def test_json_that_is_not_an_object_is_a_usage_error(value, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-censoring", type=generation_map)
    parser.add_argument("--E1", type=float_or_generation_map)
    for flag in ("--gen-censoring", "--E1"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([flag, value])
        assert exc.value.code == 2
        assert "invalid" in capsys.readouterr().err
