"""Shared CLI boilerplate for sim_ace entry points."""

from __future__ import annotations

__all__ = ["add_logging_args", "init_logging"]

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


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
    from sim_ace import setup_logging

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)
