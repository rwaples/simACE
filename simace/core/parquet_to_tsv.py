"""Convert parquet files to TSV (optionally gzipped) for use in R."""

__all__ = ["convert"]

import argparse
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


def convert(parquet_path: str, output_path: str | None = None, float_precision: int = 4, gzip: bool = True) -> None:
    """Read a parquet file and write it as a TSV.

    Args:
        parquet_path: Path to the input ``.parquet`` file.
        output_path: Path for the output file. If *None*, replaces the
            ``.parquet`` suffix with ``.tsv.gz`` (or ``.tsv`` if *gzip* is False).
        float_precision: Number of decimal places for float columns.
        gzip: Whether to gzip-compress the output.
    """
    parquet_path = str(parquet_path)
    if output_path is None:
        suffix = ".tsv.gz" if gzip else ".tsv"
        output_path = parquet_path.removesuffix(".parquet") + suffix

    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    logger.info("Read %s (%d rows, %d cols)", parquet_path, len(df), len(df.columns))

    compression = "gzip" if gzip else None
    df.to_csv(output_path, sep="\t", index=False, compression=compression, float_format=f"%.{float_precision}f")
    elapsed = time.perf_counter() - t0
    logger.info("Wrote %s (%.1fs)", output_path, elapsed)


def cli() -> None:
    """Command-line interface: parquet-to-tsv."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Convert parquet to TSV (gzipped by default)")
    add_logging_args(parser)
    parser.add_argument("parquet", nargs="+", help="Input parquet file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output path (only valid with a single input)")
    parser.add_argument("-p", "--precision", type=int, default=4, help="Decimal places for float columns (default: 4)")
    parser.add_argument("--no-gzip", action="store_true", help="Write uncompressed .tsv instead of .tsv.gz")
    args = parser.parse_args()
    init_logging(args)

    if args.output and len(args.parquet) > 1:
        parser.error("--output can only be used with a single input file")

    for path in args.parquet:
        convert(path, output_path=args.output, float_precision=args.precision, gzip=not args.no_gzip)


if __name__ == "__main__":
    cli()
