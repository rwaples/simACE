"""Linux process-group RSS sampling for pipeline benchmarks."""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.benchmark.model import COMMAND_TO_STAGE

_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


@dataclass(frozen=True)
class ProcessInfo:
    """One live process observed through procfs."""

    pid: int
    ppid: int
    pgrp: int
    rss_kb: int
    command: tuple[str, ...]


def _read_process(pid: int) -> ProcessInfo | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2 :].split()
        ppid = int(tail[1])
        pgrp = int(tail[2])
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        rss_kb = int(statm[1]) * _PAGE_KB
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    command = tuple(part.decode(errors="replace") for part in raw_command.split(b"\0") if part)
    return ProcessInfo(pid, ppid, pgrp, rss_kb, command)


def scan_process_group(pgrp: int) -> list[ProcessInfo]:
    """Return live processes whose process-group ID matches *pgrp*."""
    processes: list[ProcessInfo] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = _read_process(int(entry.name))
        if info is not None and info.pgrp == pgrp:
            processes.append(info)
    return processes


def _stage_name(info: ProcessInfo) -> str | None:
    """Return the stage a ``python -m simace <stage> ...`` process runs, or None."""
    command = info.command
    for i in range(len(command) - 2):
        if command[i] == "-m" and command[i + 1] == "simace" and command[i + 2] in COMMAND_TO_STAGE:
            return COMMAND_TO_STAGE[command[i + 2]]
    return None


def _owners(processes: list[ProcessInfo]) -> dict[int, str | None]:
    by_pid = {info.pid: info for info in processes}
    direct = {info.pid: _stage_name(info) for info in processes}
    owners: dict[int, str | None] = {}
    for info in processes:
        current = info
        seen: set[int] = set()
        owner: str | None = None
        while current.pid not in seen:
            seen.add(current.pid)
            owner = direct.get(current.pid)
            if owner is not None or current.ppid not in by_pid:
                break
            current = by_pid[current.ppid]
        owners[info.pid] = owner
    return owners


def _max_cpu_frequency_khz() -> int | None:
    values: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        try:
            values.append(int(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return max(values) if values else None


class ProcessSampler:
    """Sample one process group until stopped."""

    def __init__(self, pgrp: int, output: Path, interval_seconds: float) -> None:
        self.pgrp = pgrp
        self.output = output
        self.interval_seconds = interval_seconds
        self.peak_summed_rss_kb = 0
        self.max_individual_rss_kb = 0
        self.rule_peaks_kb: dict[str, int] = {}
        self.frequencies_khz: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rss-sampler-{pgrp}", daemon=True)

    def start(self) -> None:
        """Start the sampling thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the thread."""
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 4))

    def summary(self) -> dict[str, object]:
        """Return peak memory and CPU-frequency observations."""
        return {
            "peak_summed_rss_kb": self.peak_summed_rss_kb,
            "max_individual_rss_kb": self.max_individual_rss_kb,
            "rule_peaks_kb": dict(sorted(self.rule_peaks_kb.items())),
            "frequency_max_khz": max(self.frequencies_khz, default=None),
            "frequency_median_khz": statistics.median(self.frequencies_khz) if self.frequencies_khz else None,
        }

    def _run(self) -> None:
        started = time.monotonic()
        next_frequency = started
        with self.output.open("w", encoding="utf-8") as stream:
            while not self._stop.is_set():
                processes = scan_process_group(self.pgrp)
                owners = _owners(processes)
                total = sum(info.rss_kb for info in processes)
                largest = max((info.rss_kb for info in processes), default=0)
                by_rule: dict[str, int] = {}
                for info in processes:
                    owner = owners[info.pid]
                    if owner is not None:
                        by_rule[owner] = by_rule.get(owner, 0) + info.rss_kb
                self.peak_summed_rss_kb = max(self.peak_summed_rss_kb, total)
                self.max_individual_rss_kb = max(self.max_individual_rss_kb, largest)
                for rule, rss_kb in by_rule.items():
                    self.rule_peaks_kb[rule] = max(self.rule_peaks_kb.get(rule, 0), rss_kb)

                now = time.monotonic()
                frequency = None
                if now >= next_frequency:
                    frequency = _max_cpu_frequency_khz()
                    next_frequency = now + 2.0
                    if frequency is not None:
                        self.frequencies_khz.append(frequency)
                row = {
                    "elapsed_seconds": now - started,
                    "summed_rss_kb": total,
                    "max_individual_rss_kb": largest,
                    "frequency_max_khz": frequency,
                    "processes": [
                        {
                            "pid": info.pid,
                            "ppid": info.ppid,
                            "rss_kb": info.rss_kb,
                            "rule": owners[info.pid],
                            "command": list(info.command),
                        }
                        for info in sorted(processes, key=lambda item: item.pid)
                    ],
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                self._stop.wait(self.interval_seconds)
