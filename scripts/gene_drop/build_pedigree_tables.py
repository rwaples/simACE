r"""Build the msprime fixed-pedigree TableCollection once per rep.

The pedigree structure is identical across all 22 chromosomes, but
``msprime.PedigreeBuilder.add_individual`` is called ~600k times per chrom
in the per-chrom flow — a ~3.85s loop on a 100k-individual scenario, repeated
22 times per rep (~85s of duplicate work).

This script runs the build once and dumps the resulting ``tskit.TableCollection``
to a ``.trees`` file. Each downstream genotype_drop_chrom.py run loads it and mutates
``sequence_length`` to the per-chrom value before passing to
``msprime.sim_ancestry(initial_state=...)``. Verified that finalise is
re-runnable on the same builder and that loaded-then-mutated TCs yield
byte-identical sim_ancestry output vs an in-memory finalise.

Self-contained: inlines pedigree_to_msprime so the script runs as a plain
file. Needs the conda env in ``scripts/gene_drop/envs/tskit.yaml``. Example::

    python scripts/gene_drop/build_pedigree_tables.py \
        --pedigree results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet \
        --tables results/{folder}/{scenario}/rep{rep}/pedigree_tables.trees \
        --g-ped {G_ped} --g-pheno {G_pheno} \
        --log logs/{folder}/{scenario}/rep{rep}/build_pedigree_tables.log
"""

import argparse
import logging
import time
from pathlib import Path

import msprime
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("build_pedigree_tables")


def pedigree_to_msprime(ped: pd.DataFrame, G_ped: int, G_pheno: int, Ne: int) -> msprime.PedigreeBuilder:
    """Build an msprime PedigreeBuilder from a simACE pedigree DataFrame.

    msprime time runs backward (present = 0); simACE generation runs forward
    (founders = 0). Founders (mother == -1) get parents=None; the most recent
    G_pheno generations are marked as samples (so downstream tstrait gets
    correct dosages for every individual that will be phenotyped).
    """
    if G_pheno < 1 or G_pheno > G_ped:
        raise ValueError(f"G_pheno must be in [1, G_ped]; got G_pheno={G_pheno}, G_ped={G_ped}")
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=Ne)
    pb = msprime.PedigreeBuilder(demography=demography)
    sim_to_ms: dict[int, int] = {}

    latest_gen = G_ped - 1
    earliest_sample_gen = latest_gen - (G_pheno - 1)
    for row in ped.itertuples(index=False):
        time_ago = latest_gen - row.generation
        is_sample = row.generation >= earliest_sample_gen
        if row.mother == -1:
            parents = None
        else:
            parents = [sim_to_ms[int(row.mother)], sim_to_ms[int(row.father)]]
        ms_id = pb.add_individual(time=time_ago, is_sample=is_sample, parents=parents)
        sim_to_ms[int(row.id)] = ms_id
    return pb


def _route_logging_to_file(log_path: str | None) -> None:
    """Add a FileHandler writing to ``log_path`` when one is given."""
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main(argv: list[str] | None = None) -> None:
    """Build msprime pedigree tables from a parquet pedigree and pickle them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pedigree", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--g-ped", type=int, required=True)
    parser.add_argument("--g-pheno", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    _route_logging_to_file(args.log)
    pedigree_path = Path(args.pedigree)
    g_ped = args.g_ped
    g_pheno = args.g_pheno
    out_path = Path(args.tables)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading pedigree %s", pedigree_path)
    ped = pd.read_parquet(pedigree_path)
    n_founders = int((ped["generation"] == 0).sum())
    log.info("  pedigree: %d rows, %d founders, G_ped=%d", len(ped), n_founders, g_ped)

    log.info("building msprime pedigree (Ne=%d, G_pheno=%d)", n_founders, g_pheno)
    t = time.perf_counter()
    pb = pedigree_to_msprime(ped, G_ped=g_ped, G_pheno=g_pheno, Ne=n_founders)
    log.info("  built (%.1fs)", time.perf_counter() - t)

    # Placeholder sequence_length=1; consumer mutates per-chrom before sim_ancestry.
    tables = pb.finalise(sequence_length=1)
    tables.dump(str(out_path))
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
