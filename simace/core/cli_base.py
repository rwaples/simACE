"""Shared CLI boilerplate for simace entry points."""

from __future__ import annotations

__all__ = ["add_logging_args", "add_version_arg", "init_logging"]

import logging
from importlib.metadata import version
from typing import TYPE_CHECKING

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
