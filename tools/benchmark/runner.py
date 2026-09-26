"""Pipeline benchmark orchestration and provenance capture."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simace import __version__ as simace_version
from simace.config import resolve_defaults, resolve_scenarios
from simace.core.yaml_io import load_yaml
from tools.benchmark.model import (
    COMMAND_TO_STAGE,
    MANIFEST_SCHEMA,
    RESULTS_SCHEMA,
    SCHEMA_VERSION,
    BenchmarkError,
    build_summaries,
    write_json,
)
from tools.benchmark.process import ProcessSampler

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENVIRONMENT = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)


class CommandFailed(BenchmarkError):
    """Raised when one measured ``simace run`` command fails."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True)
class RunConfig:
    """Validated settings for one benchmark invocation."""

    folder: str
    scenarios: tuple[str, ...]
    profile: str | None
    repeats: int
    jobs: int
    cache_mode: str
    order_seed: int
    sample_interval_seconds: float
    output: Path | None
    cli_argv: tuple[str, ...]


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_text(command: list[str], *, default: str = "unavailable") -> str:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if completed.returncode == 0 and output else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cpu_info() -> tuple[str, int | None]:
    model = "unknown"
    cores: set[tuple[str, str]] = set()
    physical_id = "0"
    core_id: str | None = None
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return model, None
    for line in [*lines, ""]:
        if line.startswith("model name") and model == "unknown":
            model = line.split(":", 1)[1].strip()
        elif line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()
        elif not line.strip() and core_id is not None:
            cores.add((physical_id, core_id))
            core_id = None
    return model, len(cores) or None


def _ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _git_provenance() -> dict[str, Any]:
    status = _run_text(["git", "status", "--porcelain=v1"], default="")
    changed_files = [line[3:] for line in status.splitlines() if len(line) >= 4]
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "short_commit": _run_text(["git", "rev-parse", "--short", "HEAD"]),
        "describe": _run_text(["git", "describe", "--always", "--dirty", "--tags"]),
        "branch": _run_text(["git", "branch", "--show-current"], default="detached"),
        "dirty": bool(status),
        "changed_files": changed_files,
    }


def _scenario_sources(names: tuple[str, ...]) -> dict[str, Path]:
    wanted = set(names)
    sources: dict[str, Path] = {}
    for path in sorted((ROOT / "config").glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        raw = load_yaml(path)
        if not isinstance(raw, dict):
            continue
        for name in wanted & set(raw):
            sources[name] = path
    missing = wanted - set(sources)
    if missing:
        raise BenchmarkError(f"could not locate config source for scenarios: {sorted(missing)}")
    return sources


def _resolve_inputs(config: RunConfig) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    defaults = resolve_defaults(ROOT / "config")
    scenarios = resolve_scenarios(ROOT / "config", defaults)
    unknown = set(config.scenarios) - set(scenarios)
    if unknown:
        raise BenchmarkError(f"unknown scenarios: {sorted(unknown)}")

    resolved: dict[str, dict[str, Any]] = {}
    for name in config.scenarios:
        params = {**defaults, **scenarios[name]}
        if params["folder"] != config.folder:
            raise BenchmarkError(
                f"scenario {name!r} belongs to folder {params['folder']!r}, not requested folder {config.folder!r}"
            )
        resolved[name] = params

    sources = _scenario_sources(config.scenarios)
    paths = {ROOT / "config" / "_default.yaml", *sources.values()}
    hashes = {str(path.relative_to(ROOT)): _sha256_file(path) for path in sorted(paths)}
    return resolved, hashes


def scenario_orders(scenarios: tuple[str, ...], repeats: int, seed: int) -> tuple[list[str], list[list[str]]]:
    """Return one seeded warm-up order and rotated measured orders."""
    base = list(scenarios)
    random.Random(seed).shuffle(base)
    measured = [base[offset % len(base) :] + base[: offset % len(base)] for offset in range(repeats)]
    return base, measured


def _create_output(config: RunConfig, git: dict[str, Any]) -> Path:
    if config.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "bench-logs" / f"{stamp}-{git['short_commit']}"
    else:
        output = config.output if config.output.is_absolute() else ROOT / config.output
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BenchmarkError(f"output directory already exists: {output}") from exc
    return output.resolve()


def _build_manifest(
    config: RunConfig,
    run_id: str,
    git: dict[str, Any],
    resolved: dict[str, dict[str, Any]],
    config_hashes: dict[str, str],
    warmup_order: list[str],
    measured_orders: list[list[str]],
    output: Path,
) -> dict[str, Any]:
    cpu_model, physical_cores = _cpu_info()
    hostname = platform.node()
    pixi_lock = ROOT / "pixi.lock"
    planned = []
    if config.cache_mode == "warm":
        planned.extend(
            {"phase": "warmup", "repetition": None, "order": order, "scenario": scenario}
            for order, scenario in enumerate(warmup_order, start=1)
        )
    planned.extend(
        {"phase": "measured", "repetition": repetition, "order": order, "scenario": scenario}
        for repetition, scenarios in enumerate(measured_orders, start=1)
        for order, scenario in enumerate(scenarios, start=1)
    )
    return {
        "schema": {"name": MANIFEST_SCHEMA, "version": SCHEMA_VERSION},
        "run_id": run_id,
        "created_utc": utc_now(),
        "output_directory": str(output),
        "cli_argv": list(config.cli_argv),
        "repository": git,
        "inputs": {
            "pixi_lock_sha256": _sha256_file(pixi_lock),
            "config_files_sha256": config_hashes,
            "scenarios": {
                name: {
                    "folder": params["folder"],
                    "seed": params["seed"],
                    "N": params["N"],
                    "G_ped": params["G_ped"],
                    "parameters_sha256": _sha256_json(params),
                }
                for name, params in resolved.items()
            },
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "host_id_sha256": hashlib.sha256(hostname.encode()).hexdigest() if hostname else None,
            "cpu_model": cpu_model,
            "physical_cores": physical_cores,
            "logical_cores": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
        },
        "tools": {
            "pixi": _run_text(["pixi", "--version"]),
            "python": platform.python_version(),
            "simace": simace_version,
        },
        "execution": {
            "profile": config.profile,
            "folder": config.folder,
            "scenarios": list(config.scenarios),
            "repeats": config.repeats,
            "jobs": config.jobs,
            "cache_mode": config.cache_mode,
            "cache_root": str(output / "cache"),
            "order_seed": config.order_seed,
            "sample_interval_seconds": config.sample_interval_seconds,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
            "command": _simace_run("python", "<scenario>", config.jobs),
            "planned": planned,
        },
    }


def _cache_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _simace_run(python: str, scenario: str, jobs: int) -> list[str]:
    return [python, "-m", "simace", "run", scenario, "--force", "--jobs", str(jobs)]


def _collect_timing(folder: str, scenario: str, destination: Path, run_dir: Path) -> dict[str, dict[str, Any]]:
    """Copy the scenario's ``timing.tsv`` files into *destination* and parse their stage rows."""
    scenario_dir = ROOT / "results" / folder / scenario
    rules: dict[str, dict[str, Any]] = {}
    for source in sorted([*scenario_dir.glob("rep*/timing.tsv"), scenario_dir / "plots" / "timing.tsv"]):
        if not source.exists():
            continue
        target = destination / source.relative_to(scenario_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with target.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = COMMAND_TO_STAGE.get(row["stage"], row["stage"])
                rules.setdefault(key, {"stage": [], "peak_rss_kb": None})["stage"].append(
                    {
                        "wall_seconds": float(row["wall_s"]),
                        "max_rss_mb": float(row["max_rss_mb"]),
                        "artifact": str(target.relative_to(run_dir)),
                    }
                )
    return rules


def _parse_time_file(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                key, value = line.strip().split(": ", 1)
                values[key] = value
    try:
        rss = int(values["Maximum resident set size (kbytes)"])
    except (KeyError, ValueError):
        rss = 0
    elapsed_text = values.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    elapsed_seconds = None
    if elapsed_text is not None:
        try:
            parts = [float(part) for part in elapsed_text.split(":")]
            elapsed_seconds = sum(value * 60**power for power, value in enumerate(reversed(parts)))
        except ValueError:
            elapsed_seconds = None
    return {
        "max_rss_kb": rss,
        "cpu_percent": values.get("Percent of CPU this job got"),
        "elapsed": elapsed_text,
        "elapsed_seconds": elapsed_seconds,
    }


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_execution(
    config: RunConfig,
    run_dir: Path,
    *,
    phase: str,
    repetition: int | None,
    order: int,
    scenario: str,
) -> dict[str, Any]:
    tag = f"{phase}-r{repetition or 0:02d}-o{order:02d}-{scenario}"
    artifacts = run_dir / "runs" / tag
    artifacts.mkdir(parents=True)
    log_path = artifacts / "simace-run.log"
    time_path = artifacts / "time.txt"
    samples_path = artifacts / "processes.jsonl"

    if config.cache_mode == "warm":
        cache = run_dir / "cache" / "warm"
    else:
        cache = run_dir / "cache" / tag
    cache.mkdir(parents=True, exist_ok=True)
    cache_before = _cache_count(cache)

    command = ["/usr/bin/time", "-v", "-o", str(time_path), *_simace_run(sys.executable, scenario, config.jobs)]
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache)

    started_utc = utc_now()
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        sampler = ProcessSampler(process.pid, samples_path, config.sample_interval_seconds)
        sampler.start()
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            _terminate_group(process)
            raise
        finally:
            sampler.stop()
    monotonic_seconds = time.monotonic() - started
    memory = sampler.summary()
    rules = _collect_timing(config.folder, scenario, artifacts / "timing", run_dir)
    for rule, peak in memory["rule_peaks_kb"].items():
        rules.setdefault(rule, {"stage": [], "peak_rss_kb": None})["peak_rss_kb"] = peak
    time_result = _parse_time_file(time_path)
    # `is None`, not `or`: _parse_time_file reports an absent or unparseable
    # field as None, and GNU time prints 0:00.00 for a fast command, which
    # parses to a legitimate 0.0 the fallback must not overwrite.
    elapsed = time_result["elapsed_seconds"]
    wall_seconds = monotonic_seconds if elapsed is None else elapsed
    return {
        "phase": phase,
        "repetition": repetition,
        "order": order,
        "scenario": scenario,
        "status": "complete" if returncode == 0 else "failed",
        "returncode": returncode,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "command": command,
        "wall_seconds": wall_seconds,
        "monotonic_seconds": monotonic_seconds,
        "gnu_time_max_rss_kb": time_result["max_rss_kb"],
        "gnu_time_cpu_percent": time_result["cpu_percent"],
        "peak_summed_rss_kb": memory["peak_summed_rss_kb"],
        "max_individual_rss_kb": memory["max_individual_rss_kb"],
        "frequency_max_khz": memory["frequency_max_khz"],
        "frequency_median_khz": memory["frequency_median_khz"],
        "cache": {
            "path": str(cache.relative_to(run_dir)),
            "files_before": cache_before,
            "files_after": _cache_count(cache),
        },
        "rules": rules,
        "artifacts": {
            "log": str(log_path.relative_to(run_dir)),
            "time": str(time_path.relative_to(run_dir)),
            "process_samples": str(samples_path.relative_to(run_dir)),
        },
    }


def _preflight(config: RunConfig) -> None:
    if config.repeats < 3:
        raise BenchmarkError("repeats must be at least 3")
    if config.jobs < 1:
        raise BenchmarkError("jobs must be positive")
    if config.sample_interval_seconds <= 0:
        raise BenchmarkError("sample interval must be positive")
    if platform.system() != "Linux" or not Path("/proc").is_dir():
        raise BenchmarkError("pipeline benchmarking requires Linux procfs")
    if not Path("/usr/bin/time").is_file():
        raise BenchmarkError("pipeline benchmarking requires /usr/bin/time")


def run_benchmark(config: RunConfig) -> Path:
    """Run a complete benchmark invocation and return its new directory."""
    _preflight(config)
    git = _git_provenance()
    resolved, config_hashes = _resolve_inputs(config)
    warmup_order, measured_orders = scenario_orders(config.scenarios, config.repeats, config.order_seed)
    output = _create_output(config, git)
    run_id = output.name
    manifest = _build_manifest(config, run_id, git, resolved, config_hashes, warmup_order, measured_orders, output)
    write_json(output / "manifest.json", manifest)
    results: dict[str, Any] = {
        "schema": {"name": RESULTS_SCHEMA, "version": SCHEMA_VERSION},
        "run_id": run_id,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "executions": [],
        "summaries": [],
        "error": None,
    }
    write_json(output / "results.json", results)

    planned = manifest["execution"]["planned"]
    try:
        for item in planned:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {item['phase']} "
                f"{item['scenario']} repetition={item['repetition']} order={item['order']}",
                flush=True,
            )
            execution = _run_execution(config, output, **item)
            results["executions"].append(execution)
            results["summaries"] = build_summaries(results["executions"])
            write_json(output / "results.json", results)
            if execution["returncode"] != 0:
                raise CommandFailed(
                    f"benchmark command failed for {item['scenario']} with exit {execution['returncode']}; "
                    f"see {output / execution['artifacts']['log']}",
                    execution["returncode"],
                )
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        results["error"] = {"type": "KeyboardInterrupt", "message": "benchmark interrupted"}
        results["finished_utc"] = utc_now()
        results["summaries"] = build_summaries(results["executions"])
        write_json(output / "results.json", results)
        raise
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = {"type": type(exc).__name__, "message": str(exc)}
        results["finished_utc"] = utc_now()
        results["summaries"] = build_summaries(results["executions"])
        write_json(output / "results.json", results)
        raise

    results["status"] = "complete"
    results["finished_utc"] = utc_now()
    results["summaries"] = build_summaries(results["executions"])
    write_json(output / "results.json", results)
    return output
