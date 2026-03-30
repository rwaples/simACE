"""Convert parquet files to gzipped TSV for use in R."""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


def convert(parquet_path: str, output_path: str | None = None, float_precision: int = 4) -> None:
    """Read a parquet file and write it as a gzipped TSV.

    Parameters
    ----------
    parquet_path : path to the input .parquet file
    output_path : path for the output .tsv.gz file.  If *None*, replaces
        the ``.parquet`` suffix with ``.tsv.gz``.
    float_precision : number of decimal places for float columns (default 4)
    """
    parquet_path = str(parquet_path)
    if output_path is None:
        output_path = parquet_path.replace(".parquet", ".tsv.gz")

    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    logger.info("Read %s (%d rows, %d cols)", parquet_path, len(df), len(df.columns))

    df.to_csv(output_path, sep="\t", index=False, compression="gzip", float_format=f"%.{float_precision}f")
    elapsed = time.perf_counter() - t0
    logger.info("Wrote %s (%.1fs)", output_path, elapsed)


def cli() -> None:
    """Command-line interface: parquet-to-tsv."""
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Convert parquet to gzipped TSV")
    add_logging_args(parser)
    parser.add_argument("parquet", nargs="+", help="Input parquet file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output path (only valid with a single input)")
    parser.add_argument(
        "-p", "--precision", type=int, default=4, help="Decimal places for float columns (default: 4)"
    )
    args = parser.parse_args()
    init_logging(args)

    if args.output and len(args.parquet) > 1:
        parser.error("--output can only be used with a single input file")

    for path in args.parquet:
        convert(path, output_path=args.output, float_precision=args.precision)


if __name__ == "__main__":
    cli()
