"""Fingerprint what the shipped scenarios compute, to gate refactors that must not change it.

Two modes, each writing one JSON file:

``decisions``
    Every shipped scenario (``config/*.yaml``), no simulation: the stage argv
    of rep 1, its ``params.yaml`` dict (``simace_version`` dropped), and every
    derived decision the analysis takes from it: the assortment schedule and
    span, the per-trait AM mode, the analytic gate, the reference lines and
    the effective-size expectations.

``pipeline``
    A covering subset of scenarios run end to end at small N in a scratch
    root (``simace run`` then ``simace validate``): each rep's pedigree
    content hash, its ``report.yaml`` and ``validation.yaml`` with timing and
    version fields dropped.

``bytes``
    A covering subset as shipped (ascertainment kept) plus variants of
    ``small_test`` that reach the report's empty and early-return branches,
    run with plots: each rep's ``report.yaml``, ``plot_payload.yaml`` and
    figures, byte for byte, with version and date lines masked.

Compare two fingerprints with ``diff`` (the files are sorted, indented JSON;
``diff -r`` for bytes)::

    pixi run python tools/scenario_fingerprint.py decisions -o before.json
    pixi run python tools/scenario_fingerprint.py pipeline --scratch /tmp/fp -o before-pipeline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Scenarios the pipeline mode runs: every assortment-matrix, Wright-Fisher,
#: per-generation assortment and per-generation E path, C transmission with
#: and without AM, single- and both-trait AM, and the smoke test.
PIPELINE_SUBSET = (
    "am",
    "am_asym",
    "am_plus_c",
    "inh_am_matrix",
    "adhd_composite_v1",
    "adhd_composite_v2",
    "bench1M_am",
    "wf_smoke_short",
    "wf_smoke_long",
    "maternal",
    "both_parents",
    "maternal_am",
    "inh_ctm",
    "growing_AM",
    "inh_am_mixed",
    "am_per_generation",
    "growing_E",
    "e_flat_pergen",
    "am_t1",
    "assortative_mating_both",
    "small_test",
)
PIPELINE_N = 2000

#: Report and validation keys that vary between builds or runs, not between parameter readings.
_VOLATILE = {"simace_version", "version", "source", "git", "elapsed", "elapsed_s", "seconds", "timing", "created"}


def _canon(value: Any) -> Any:
    """JSON-safe, order-stable form: floats rounded to 12 significant digits, NaN as a string."""
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])) if k not in _VOLATILE}
    if isinstance(value, list | tuple):
        return [_canon(v) for v in value]
    if hasattr(value, "tolist"):
        return _canon(value.tolist())
    if isinstance(value, bool) or value is None or isinstance(value, int | str):
        return value
    if isinstance(value, float):
        return "nan" if math.isnan(value) else float(f"{value:.12g}")
    return repr(value)


def _decisions(params: dict[str, Any]) -> dict[str, Any]:
    """Every derived decision analysis takes from a rep's ``params.yaml``.

    Update the calls here when signatures change; the values must not.
    """
    from simace.analysis.quality.am_relatedness import am_relatedness_mode
    from simace.analysis.stats.effective_size import theoretical_expectations
    from simace.core import relationships as rel
    from simace.simulation.params import SimulationParams

    sim = SimulationParams.from_mapping(params)
    records = (sim.assortment, sim.c_transmission, sim.vc)
    out: dict[str, Any] = {
        "schedule": sim.assortment.schedule,
        "span": (sim.span.G_sim, sim.span.burnin),
        "mated_matrix": sim.assortment.mated_matrix,
        "effective_size": theoretical_expectations(sim),
    }
    for t in (1, 2):
        out[f"t{t}"] = {
            "am_mode": am_relatedness_mode(sim.assortment, t),
            "closed_form": rel.closed_form_holds(sim.assortment, sim.vc, t),
            "midparent_slope": rel.reference_midparent_slope(*records, t, truth=-99.0),
            "types": {
                ptype: {
                    "analytic": rel.analytic_applies(ptype, *records, t),
                    "reference": rel.reference_corr(ptype, *records, t, truth={ptype: {"r": -99.0}}),
                }
                for ptype in rel.RELATIONSHIP_TYPES
            },
        }
    return out


def decisions(out: Path) -> None:
    """Write the no-simulation fingerprint of every shipped scenario to ``out``."""
    from simace.cli.layout import Layout
    from simace.cli.stages import PARAMS_YAML_KEYS, STAGES, ResolvedRep
    from simace.config import load_scenarios
    from simace.core.yaml_io import dump_yaml, load_yaml
    from simace.simulation.emit_params import emit_params

    layout = Layout(root=Path("results"), logs=Path("logs"))
    result: dict[str, Any] = {}
    tmp = out.with_suffix(".params.yaml")
    for name, scenario in sorted(load_scenarios(ROOT / "config").items()):
        params = scenario.flat
        rep = ResolvedRep(scenario.folder, name, 1, params)
        emitted = emit_params(seed=rep.seed, rep=1, **{key: params[key] for key in PARAMS_YAML_KEYS})
        dump_yaml(emitted, tmp, sort_keys=True)
        on_disk = load_yaml(tmp)
        entry: dict[str, Any] = {
            "argv": {stage.name: stage.argv(rep, layout) for stage in STAGES},
            "params_yaml": {k: v for k, v in on_disk.items() if k != "simace_version"},
        }
        try:
            entry["decisions"] = _decisions(on_disk)
        except Exception as exc:  # a decision that raises is part of the fingerprint
            entry["decisions"] = f"{type(exc).__name__}: {exc}"
        result[name] = entry
    tmp.unlink(missing_ok=True)
    out.write_text(json.dumps(_canon(result), indent=1, sort_keys=True))
    print(f"{len(result)} scenarios -> {out}")


def _scratch_config(scratch: Path) -> None:
    """Copy ``_default.yaml`` and write ``fp.yaml``: the subset's raw entries at N = PIPELINE_N, one rep."""
    from simace.core.yaml_io import load_yaml

    config = scratch / "config"
    config.mkdir(parents=True, exist_ok=True)
    (scratch / "pyproject.toml").write_text('[project]\nname = "simace"\n')
    (config / "_default.yaml").write_text((ROOT / "config" / "_default.yaml").read_text())
    raw: dict[str, Any] = {}
    for path in sorted((ROOT / "config").glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        for name, source in (load_yaml(path) or {}).items():
            if name in PIPELINE_SUBSET:
                entry = dict(source)
                entry.pop("folder", None)
                entry["N"] = min(int(entry.get("N", PIPELINE_N)), PIPELINE_N)
                entry["replicates"] = 1
                entry.pop("ascertainment", None)
                entry.pop("N_sample", None)
                raw[name] = entry
    missing = set(PIPELINE_SUBSET) - set(raw)
    if missing:
        raise SystemExit(f"subset scenarios not found: {sorted(missing)}")
    (config / "fp.yaml").write_text(yaml.safe_dump(raw, sort_keys=True))


def _content_hash(pedigree: Path) -> str:
    from simace.core.parquet import load_parquet

    frame = load_parquet(pedigree)
    return hashlib.sha256(frame.write_csv().encode()).hexdigest()


def pipeline(scratch: Path, out: Path, jobs: int) -> None:
    """Run the subset end to end under ``scratch`` and write its fingerprint to ``out``."""
    from simace.core.yaml_io import load_yaml

    _scratch_config(scratch)
    roots = ["--config-dir", str(scratch / "config"), "--results", str(scratch / "results")]
    roots += ["--logs", str(scratch / "logs")]
    for cmd in (["run", "fp", "--no-plots"], ["validate", "fp", "--no-plots"]):
        done = subprocess.run(["simace", *cmd, "--jobs", str(jobs), *roots], cwd=scratch, check=False)
        if done.returncode not in (0, 1):  # validate exits 1 on a failing verdict, which is fingerprinted
            raise SystemExit(f"simace {cmd[0]} exited {done.returncode}")
    result: dict[str, Any] = {}
    for rep_dir in sorted((scratch / "results" / "fp").glob("*/rep1")):
        entry: dict[str, Any] = {"pedigree": _content_hash(rep_dir / "pedigree.parquet")}
        for name in ("report.yaml", "validation.yaml"):
            path = rep_dir / name
            entry[name] = load_yaml(path) if path.exists() else None
        result[rep_dir.parent.name] = entry
    out.write_text(json.dumps(_canon(result), indent=1, sort_keys=True))
    print(f"{len(result)} reps -> {out}")


#: Scenarios the bytes mode runs as shipped (N capped at PIPELINE_N, one rep, ascertainment kept).
BYTES_SUBSET = (
    "small_test",
    "coverage_scenario",
    "am",
    "am_asym",
    "am_plus_c",
    "inh_am_matrix",
    "wf_smoke_short",
    "maternal",
    "inh_ctm",
    "am_per_generation",
    "growing_E",
)

#: Variants of small_test that reach the report's empty and early-return branches.
BYTES_VARIANTS: dict[str, dict[str, Any]] = {
    "br_ascertained": {"ascertainment": {"N_sample": 300, "case_ascertainment_ratio": 3, "dropout_rate": 0.1}},
    "br_no_twins": {"pedigree": {"p_mztwin": 0.0}},
    "br_tiny": {"N": 30, "G_ped": 3, "G_sim": 3, "G_pheno": 2},
    "br_one_generation": {"G_ped": 1, "G_sim": 1, "G_pheno": 1},
    "br_last_generation": {"G_pheno": 1, "ascertainment": {"N_sample": 200}},
    "br_zero_windows": {"censoring": {"gen_censoring": {g: [80, 80] for g in range(6)}}},
    "br_no_windows": {"censoring": {"gen_censoring": {}}},
    "br_zero_A": {"pedigree": {"trait1": {"A": 0.0, "C": 0.0, "E": 1.0}}},
}

#: Lines that differ between builds, not between computations.
_VOLATILE_LINE = ("simace_version:", "<dc:date>")


def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = (
            _deep_update(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def _bytes_config(scratch: Path) -> None:
    """Write ``fp.yaml``: the bytes subset as shipped (one rep, N <= PIPELINE_N) plus the branch variants."""
    from simace.core.yaml_io import load_yaml

    config = scratch / "config"
    config.mkdir(parents=True, exist_ok=True)
    (scratch / "pyproject.toml").write_text('[project]\nname = "simace"\n')
    (config / "_default.yaml").write_text((ROOT / "config" / "_default.yaml").read_text())
    raw: dict[str, Any] = {}
    for path in sorted((ROOT / "config").glob("*.yaml")):
        if not path.name.startswith("_"):
            raw.update({k: v for k, v in (load_yaml(path) or {}).items() if k in BYTES_SUBSET})
    missing = set(BYTES_SUBSET) - set(raw)
    if missing:
        raise SystemExit(f"subset scenarios not found: {sorted(missing)}")
    for name, source in list(raw.items()):
        entry = {k: v for k, v in source.items() if k != "folder"}
        entry["N"] = min(int(entry.get("N", PIPELINE_N)), PIPELINE_N)
        entry["replicates"] = 1
        raw[name] = entry
    for name, extra in BYTES_VARIANTS.items():
        raw[name] = _deep_update(raw["small_test"], extra)
    (config / "fp.yaml").write_text(yaml.safe_dump(raw, sort_keys=True))


def _masked(path: Path) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    return b"".join(line for line in lines if not any(tag.encode() in line for tag in _VOLATILE_LINE))


def bytes_mode(scratch: Path, out: Path, jobs: int) -> None:
    """Run the bytes subset with plots under ``scratch``; copy each rep's report, payload and figures, masked, to ``out``.

    Compare two runs with ``diff -r before/ after/``; ``out/SHA256SUMS`` lists every file.
    """
    out.mkdir(parents=True, exist_ok=True)
    _bytes_config(scratch)
    roots = ["--config-dir", str(scratch / "config"), "--results", str(scratch / "results")]
    roots += ["--logs", str(scratch / "logs")]
    done = subprocess.run(["simace", "run", "fp", "--jobs", str(jobs), *roots], cwd=scratch, check=False)
    if done.returncode not in (0, 1):  # 1: a scenario's plots failed; its failed plot logs are fingerprinted
        raise SystemExit(f"simace run exited {done.returncode}")
    results = scratch / "results" / "fp"
    failed = sorted(
        p.parent.name for p in (scratch / "logs" / "fp").glob("*/plot.log") if b"Traceback" in p.read_bytes()
    )
    (out / "PLOT_FAILURES").write_text("".join(f"{name}\n" for name in failed))
    files = [p for p in sorted(results.glob("*/rep1/*.yaml")) if p.name in ("report.yaml", "plot_payload.yaml")]
    # plots.yaml and timing.tsv are run bookkeeping (input mtimes, timings), not figures.
    bookkeeping = {"plots.yaml", "timing.tsv"}
    files += [
        p for p in sorted(results.glob("*/plots/*")) if p.is_file() and p.suffix != ".pdf" and p.name not in bookkeeping
    ]
    sums = []
    for path in files:
        rel = path.relative_to(results)
        data = _masked(path)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {rel}\n")
    (out / "SHA256SUMS").write_text("".join(sums))
    print(f"{len(sums)} files from {len(BYTES_SUBSET) + len(BYTES_VARIANTS)} scenarios -> {out}")


def main() -> None:
    """Parse the mode and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("decisions", help="All shipped scenarios, no simulation")
    d.add_argument("-o", "--output", type=Path, required=True)
    p = sub.add_parser("pipeline", help="The covering subset, run end to end at small N")
    p.add_argument("--scratch", type=Path, required=True, help="Empty scratch root for config/results/logs")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    b = sub.add_parser("bytes", help="Report, payload and figure bytes of a covering subset plus branch variants")
    b.add_argument("--scratch", type=Path, required=True, help="Empty scratch root for config/results/logs")
    b.add_argument("-o", "--output", type=Path, required=True, help="Directory for the masked copies")
    b.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.mode == "decisions":
        decisions(args.output)
    elif args.mode == "pipeline":
        pipeline(args.scratch, args.output, args.jobs)
    else:
        bytes_mode(args.scratch, args.output, args.jobs)


if __name__ == "__main__":
    main()
