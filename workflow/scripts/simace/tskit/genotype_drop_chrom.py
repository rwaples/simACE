"""Snakemake wrapper: per-chromosome pedigree drop + ancestry graft.

Self-contained: inlines pedigree_to_msprime, drop_one_chromosome, and
graft_ancestry from external/tskit/run_trial_graft.py (which is gitignored).
Recombination uses the per-chromosome SimHumanity HapMapII_GRCh38 rate map.
Runs in the workflow's tskit conda env via `--use-conda`.
"""

import logging
import time
from pathlib import Path

import msprime
import numpy as np
import pandas as pd
import tskit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("genotype_drop")


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


def drop_one_chromosome(
    pedigree_tables: tskit.TableCollection,
    recombination: msprime.RateMap | float,
    seed: int,
) -> tskit.TreeSequence:
    """Run one msprime fixed_pedigree drop. No mutations added at this step."""
    return msprime.sim_ancestry(
        initial_state=pedigree_tables,
        model="fixed_pedigree",
        recombination_rate=recombination,
        random_seed=seed,
    )


def graft_ancestry(
    ts_drop: tskit.TreeSequence,
    ts_anc: tskit.TreeSequence,
    n_founders: int,
    founder_time: int,
    chosen_inds: np.ndarray,
) -> tskit.TreeSequence:
    """Replace recapitate by grafting ts_anc onto ts_drop's founder lineages.

    The first 2*n_founders nodes of ts_drop are the founder haplotype nodes
    (PedigreeBuilder allocates 2 nodes per individual in DataFrame order;
    simACE writes founders first). They get unioned 1:1 with the canonical
    sample nodes 0..2N-1 of ts_anc after simplifying ts_anc to those founders.
    """
    n_sample_nodes = 2 * n_founders
    chosen = np.asarray(chosen_inds)
    if len(chosen) != n_founders:
        raise ValueError(f"chosen_inds has {len(chosen)} entries, expected {n_founders}")

    t0 = time.perf_counter()
    sample_nodes: list[int] = []
    for ind_id in chosen:
        sample_nodes.extend(int(n) for n in ts_anc.individual(int(ind_id)).nodes)
    log.info("    graft.sample_nodes_build: %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    ts_anc_sub = ts_anc.simplify(samples=sample_nodes, filter_populations=False)
    log.info(
        "    graft.simplify_anc: %.2fs (%d -> %d nodes, %d -> %d edges, %d -> %d sites)",
        time.perf_counter() - t0,
        ts_anc.num_nodes,
        ts_anc_sub.num_nodes,
        ts_anc.num_edges,
        ts_anc_sub.num_edges,
        ts_anc.num_sites,
        ts_anc_sub.num_sites,
    )

    t0 = time.perf_counter()
    tables_anc = ts_anc_sub.dump_tables()
    tables_anc.nodes.set_columns(
        flags=tables_anc.nodes.flags,
        time=tables_anc.nodes.time + founder_time,
        population=tables_anc.nodes.population,
        individual=tables_anc.nodes.individual,
        metadata=tables_anc.nodes.metadata,
        metadata_offset=tables_anc.nodes.metadata_offset,
    )
    mut_time = tables_anc.mutations.time
    known = ~np.isnan(mut_time)
    if known.any():
        new_mut_time = mut_time.copy()
        new_mut_time[known] = mut_time[known] + founder_time
        tables_anc.mutations.set_columns(
            site=tables_anc.mutations.site,
            node=tables_anc.mutations.node,
            time=new_mut_time,
            derived_state=tables_anc.mutations.derived_state,
            derived_state_offset=tables_anc.mutations.derived_state_offset,
            parent=tables_anc.mutations.parent,
            metadata=tables_anc.mutations.metadata,
            metadata_offset=tables_anc.mutations.metadata_offset,
        )
    log.info("    graft.time_shift: %.2fs", time.perf_counter() - t0)

    node_mapping = np.full(tables_anc.nodes.num_rows, tskit.NULL, dtype=np.int32)
    node_mapping[:n_sample_nodes] = np.arange(n_sample_nodes, dtype=np.int32)

    t0 = time.perf_counter()
    tables_drop = ts_drop.dump_tables()
    log.info("    graft.dump_drop_tables: %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    # Founder nodes in ts_drop are population 0; matched ts_anc samples are p2.
    # check_shared_equality=False accepts the population mismatch.
    tables_drop.union(tables_anc, node_mapping, check_shared_equality=False)
    log.info(
        "    graft.union: %.2fs (drop nodes=%d edges=%d, anc nodes=%d edges=%d sites=%d)",
        time.perf_counter() - t0,
        ts_drop.num_nodes,
        ts_drop.num_edges,
        tables_anc.nodes.num_rows,
        tables_anc.edges.num_rows,
        tables_anc.sites.num_rows,
    )

    t0 = time.perf_counter()
    tables_drop.sort()
    log.info("    graft.sort: %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    ts_out = tables_drop.tree_sequence()
    log.info("    graft.tree_sequence: %.2fs", time.perf_counter() - t0)
    return ts_out


def load_simhumanity_rate_map(simhumanity_dir: Path, chrom_n: int, expected_seqlen: int) -> msprime.RateMap:
    """Load the SimHumanity HapMapII_GRCh38 rate map for one chromosome.

    File format: SLiM-style `(end_position, rate)` rows; first row is
    `0,0.0` (degenerate length-0 interval). The map is extended with the
    last rate to reach `expected_seqlen` if it falls short.
    """
    base = simhumanity_dir / "stdpopsim extraction" / "extracted"
    length_path = base / f"chr{chrom_n}_length.txt"
    recomb_path = base / f"chr{chrom_n}_recombination.txt"
    if not recomb_path.exists():
        raise FileNotFoundError(
            f"SimHumanity recomb file missing: {recomb_path}. Run "
            "`git -C external/SimHumanity checkout main` to populate the working tree."
        )
    declared_len = int(length_path.read_text().strip())
    if int(expected_seqlen) != declared_len:
        raise ValueError(f"chr{chrom_n}: ancestry seqlen={expected_seqlen} != SimHumanity chr length={declared_len}")

    positions: list[int] = []
    rates: list[float] = []
    for line in recomb_path.read_text().strip().splitlines():
        p_str, r_str = line.split(",")
        positions.append(int(p_str))
        rates.append(float(r_str))
    if positions[0] != 0:
        raise ValueError(f"chr{chrom_n}: recomb map first position {positions[0]} != 0")
    if positions[-1] > declared_len:
        raise ValueError(f"chr{chrom_n}: recomb map ends at {positions[-1]} > declared length {declared_len}")
    if positions[-1] < declared_len:
        positions.append(declared_len)
        rates.append(rates[-1])
    # File row i = (end of interval i, rate within interval i ending at that pos).
    # Row 0 is degenerate (0, 0.0). For msprime.RateMap we want
    # rate[i] == rate inside [position[i], position[i+1]); since position[0]==0,
    # the file's rates[1:] line up with msprime's rate array.
    return msprime.RateMap(position=positions, rate=rates[1:])


def chosen_inds_from_canonical(ts_anc: tskit.TreeSequence) -> np.ndarray:
    """All p2 individuals in canonical sample-node order (one per pair of nodes)."""
    samples = np.asarray(ts_anc.samples())
    expected = np.arange(len(samples), dtype=samples.dtype)
    if not np.array_equal(samples, expected):
        raise RuntimeError(
            f"ts_anc samples not in canonical 0..{len(samples) - 1} order; input must be a canonicalized .trees file"
        )
    node_inds = ts_anc.tables.nodes.individual[samples]
    return node_inds[::2].astype(np.int64)


def _route_logging_to_snakemake_log() -> None:
    """Add a FileHandler that writes to snakemake.log[0].

    Snakemake's `script:` rule runner exposes the rule's `log:` path as
    snakemake.log[0] but doesn't auto-redirect Python stderr to it (only
    `shell:` rules get that redirection). Without this, all `log.info(...)`
    output goes to Snakemake's central log instead of the per-rule file.
    """
    log_path = snakemake.log[0] if snakemake.log else None
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    """Snakemake entry: pedigree drop + ancestry graft for one chromosome."""
    _route_logging_to_snakemake_log()
    pedigree_path = Path(snakemake.input.pedigree)
    trees_path = Path(snakemake.input.trees)
    simhumanity_dir = Path(snakemake.params.simhumanity_dir)
    g_ped = int(snakemake.params.G_ped)
    g_pheno = int(snakemake.params.G_pheno)
    seed = int(snakemake.params.seed)
    chrom_n = int(snakemake.wildcards.n)
    out_path = Path(snakemake.output.trees)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading pedigree %s", pedigree_path)
    ped = pd.read_parquet(pedigree_path)
    n_founders = int((ped["generation"] == 0).sum())
    log.info("  pedigree: %d rows, %d founders, G_ped=%d", len(ped), n_founders, g_ped)

    log.info("loading ancestry %s", trees_path)
    t = time.perf_counter()
    ts_anc = tskit.load(str(trees_path))
    n_eligible = ts_anc.num_samples // 2
    log.info(
        "  ts_anc: seqlen=%.1f Mb, %d samples (= %d inds), %d sites, %d muts (load %.1fs)",
        ts_anc.sequence_length / 1e6,
        ts_anc.num_samples,
        n_eligible,
        ts_anc.num_sites,
        ts_anc.num_mutations,
        time.perf_counter() - t,
    )
    if n_founders != n_eligible:
        raise ValueError(
            f"chr{chrom_n}: pedigree has {n_founders} founders but ts_anc has "
            f"{n_eligible} canonical individuals — config N must equal n_eligible_p2"
        )

    chosen_inds = chosen_inds_from_canonical(ts_anc)
    log.info("  chosen_inds: %d (first 10: %s)", len(chosen_inds), chosen_inds[:10].tolist())

    log.info("loading recomb map for chr%d", chrom_n)
    rate_map = load_simhumanity_rate_map(simhumanity_dir, chrom_n, int(ts_anc.sequence_length))
    log.info(
        "  rate map: %d intervals, mean rate %.3e per bp/gen",
        len(rate_map.rate),
        float(rate_map.mean_rate),
    )

    log.info("building msprime pedigree (Ne=%d, G_pheno=%d)", n_founders, g_pheno)
    t = time.perf_counter()
    pb = pedigree_to_msprime(ped, G_ped=g_ped, G_pheno=g_pheno, Ne=n_founders)
    pedigree_tables = pb.finalise(sequence_length=ts_anc.sequence_length)
    log.info("  built (%.1fs)", time.perf_counter() - t)

    log.info("dropping pedigree (seed=%d)", seed)
    t = time.perf_counter()
    ts_drop = drop_one_chromosome(pedigree_tables, rate_map, seed)
    log.info(
        "  dropped: %d trees, %d edges (%.1fs)",
        ts_drop.num_trees,
        ts_drop.num_edges,
        time.perf_counter() - t,
    )

    log.info("grafting ancestry")
    t = time.perf_counter()
    founder_time = g_ped - 1
    ts_grafted = graft_ancestry(
        ts_drop,
        ts_anc,
        n_founders=n_founders,
        founder_time=founder_time,
        chosen_inds=chosen_inds,
    )
    log.info(
        "  grafted: %d trees, %d edges, %d sites, %d muts (%.1fs)",
        ts_grafted.num_trees,
        ts_grafted.num_edges,
        ts_grafted.num_sites,
        ts_grafted.num_mutations,
        time.perf_counter() - t,
    )

    ts_grafted.dump(str(out_path))
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
