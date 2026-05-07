"""Snakemake wrapper: verify the canonicalized concat output.

Two checks:
  1. Sidecar self-consistency: recompute the BLAKE2b fingerprint of the
     concat tables and compare to the sidecar `<trees>.fingerprint` written
     by the concat rule. Detects on-disk corruption / incomplete writes.
  2. Golden expectations: read `tests/data/tskit_preprocess_fingerprint.json`
     and assert n_eligible_individuals, num_chromosomes, total_seqlen, and
     per-chrom (n_samples, n_sites, n_muts) match. Detects source-data drift.

Bootstrap: set env var `SIMACE_TSKIT_WRITE_GOLDEN=1` to write the current
summary into the golden file instead of asserting against it. Use this on
the first ever run (or after a deliberate source-data update), then commit.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import tskit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("verify")


_HASH_TABLES = (
    ("nodes", ("flags", "time", "population", "individual")),
    ("edges", ("left", "right", "parent", "child")),
    ("sites", ("position",)),
    ("mutations", ("site", "node", "parent", "time")),
    ("individuals", ("flags",)),
)


def fingerprint_tables(ts: tskit.TreeSequence) -> str:
    """Recompute the BLAKE2b-128 hash used by the concat sidecar."""
    h = hashlib.blake2b(digest_size=16)
    tables = ts.tables
    for table_name, cols in _HASH_TABLES:
        table = getattr(tables, table_name)
        h.update(table_name.encode())
        h.update(int(table.num_rows).to_bytes(8, "little"))
        for col in cols:
            arr = np.ascontiguousarray(getattr(table, col))
            h.update(col.encode())
            h.update(arr.tobytes())
    h.update(b"sequence_length")
    h.update(np.float64(ts.sequence_length).tobytes())
    return h.hexdigest()


def golden_from_summary(summary: dict) -> dict:
    """Build the canonical golden dict from a preprocess summary."""
    return {
        "n_eligible_individuals": int(summary["consistency"]["n_eligible_individuals"]),
        "num_chromosomes": int(summary["n_chromosomes"]),
        "total_seqlen": float(summary["totals"]["total_seqlen"]),
        "per_chromosome": [
            {
                "name": r["name"],
                "n_samples": int(r["num_samples_post"]),
                "n_sites": int(r["num_sites_post"]),
                "n_muts": int(r["num_mutations_post"]),
            }
            for r in summary["per_chromosome"]
        ],
    }


def assert_golden(observed: dict, golden: dict) -> None:
    """Diff observed against the checked-in golden; raise if anything mismatches."""
    diffs: list[str] = [
        f"{k}: observed={observed[k]} golden={golden[k]}"
        for k in ("n_eligible_individuals", "num_chromosomes")
        if observed[k] != golden[k]
    ]
    if not np.isclose(observed["total_seqlen"], golden["total_seqlen"]):
        diffs.append(f"total_seqlen: observed={observed['total_seqlen']:.3f} golden={golden['total_seqlen']:.3f}")
    obs_pc = {r["name"]: r for r in observed["per_chromosome"]}
    gold_pc = {r["name"]: r for r in golden["per_chromosome"]}
    if set(obs_pc) != set(gold_pc):
        diffs.append(f"chromosome set differs: observed={sorted(obs_pc)} golden={sorted(gold_pc)}")
    diffs.extend(
        f"{name}.{k}: observed={obs_pc[name][k]} golden={gold_pc[name][k]}"
        for name in sorted(set(obs_pc) & set(gold_pc))
        for k in ("n_samples", "n_sites", "n_muts")
        if obs_pc[name][k] != gold_pc[name][k]
    )
    if diffs:
        raise AssertionError(
            "tskit_preprocess golden mismatch (set SIMACE_TSKIT_WRITE_GOLDEN=1 to update):\n  " + "\n  ".join(diffs)
        )


def _route_logging_to_snakemake_log() -> None:
    """Route logging to snakemake.log[0].

    Snakemake's `script:` runner exposes the rule's log path via
    snakemake.log[0] but doesn't auto-redirect Python stderr to it.
    """
    log_path = snakemake.log[0] if snakemake.log else None
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    """Snakemake entry: golden + sidecar fingerprint checks."""
    _route_logging_to_snakemake_log()
    trees_path = Path(snakemake.input.trees)
    fp_path = Path(snakemake.input.fingerprint)
    summary_path = Path(snakemake.input.summary)
    golden_path = Path(snakemake.params.golden)

    summary = json.loads(summary_path.read_text())
    observed = golden_from_summary(summary)

    if os.environ.get("SIMACE_TSKIT_WRITE_GOLDEN") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        log.warning("WROTE GOLDEN to %s (SIMACE_TSKIT_WRITE_GOLDEN=1). Commit it.", golden_path)
    else:
        if not golden_path.exists():
            raise FileNotFoundError(
                f"golden file {golden_path} missing — run once with SIMACE_TSKIT_WRITE_GOLDEN=1 "
                "to bootstrap, then commit."
            )
        golden = json.loads(golden_path.read_text())
        assert_golden(observed, golden)
        log.info(
            "golden check passed (n_eligible=%d, n_chrom=%d, total_seqlen=%.1f Mb)",
            observed["n_eligible_individuals"],
            observed["num_chromosomes"],
            observed["total_seqlen"] / 1e6,
        )

    sidecar_hex = fp_path.read_text().strip()
    log.info("recomputing fingerprint from %s", trees_path)
    ts_all = tskit.load(str(trees_path))
    recomputed = fingerprint_tables(ts_all)
    if recomputed != sidecar_hex:
        raise AssertionError(f"sidecar fingerprint mismatch: sidecar={sidecar_hex} recomputed={recomputed}")
    log.info("sidecar fingerprint check passed (%s)", sidecar_hex)


if __name__ == "__main__":
    main()
