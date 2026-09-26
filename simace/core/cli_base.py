"""Shared CLI boilerplate for simace entry points."""

from __future__ import annotations

__all__ = [
    "add_logging_args",
    "add_version_arg",
    "float_or_generation_map",
    "generation_map",
    "init_logging",
    "yaml_mapping",
]

import json
import logging
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse


def add_version_arg(parser: argparse.ArgumentParser, dist: str) -> None:
    """Add a ``--version`` action printing the installed distribution version.

    Args:
        parser: Argument parser to add the version flag to.
        dist: Distribution name to query via ``importlib.metadata.version``
            (e.g. ``"simace"``, ``"fitace"``, ``"fitace_epimight"``).
    """
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version(dist)}",
    )


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add standard -v/--verbose and -q/--quiet arguments.

    Args:
        parser: Argument parser to add logging flags to.
    """
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")


def init_logging(args: argparse.Namespace) -> None:
    """Derive log level from parsed args and call ``setup_logging()``.

    Args:
        args: Parsed namespace containing ``verbose`` and ``quiet`` flags.
    """
    from simace import setup_logging

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)


def generation_map(value: str) -> dict[int, Any]:
    """Parse a JSON object keyed by generation into a dict with ``int`` keys.

    JSON object keys are always strings; the domain functions expect the
    integer generation keys that config loading produces.
    """
    return {int(k): v for k, v in json.loads(value).items()}


def float_or_generation_map(value: str) -> float | dict[int, Any]:
    """Parse a scalar, or a JSON per-generation map such as ``'{"0": 0.5, "4": 0.7}'``."""
    try:
        return float(value)
    except ValueError:
        return generation_map(value)


def yaml_mapping(value: str) -> dict[Any, Any]:
    """Parse a YAML (or JSON) flow mapping such as ``'{distribution: weibull, scale: 2160}'``.

    YAML keeps integer keys as integers, so nested per-generation maps
    survive the round trip that JSON would stringify.
    """
    import yaml

    parsed = yaml.safe_load(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a mapping, got {value!r}")
    return parsed
