#!/usr/bin/env python
"""Say whether this host is quiet enough to time code on, and why not.

Wall times on this laptop have come out 3-7x high from a clamped clock and from
other sessions' jobs, while every governor setting read fine. This checks, in
one call:

- the clock under load: one core busy-loops for a second, then its MHz is read
  (PROCHOT pins it near 800 MHz);
- the clamps: ``platform_profile`` ``quiet``, ``intel_pstate/no_turbo``, and a
  ``scaling_max_freq`` below ``cpuinfo_max_freq`` (system76-power sets 2600 MHz);
- the load: 1-minute load average above ``--max-load``, and the busiest
  processes that are not this one.

Prints one line per finding and exits 1 if any check fails, 0 if the host is
quiet. ``--wait SECONDS`` polls until the load settles (clock clamps fail at
once). Fixes for each clamp are in the memory note on checking the CPU clock.

    pixi run python tools/host_check.py
    pixi run python tools/host_check.py --wait 600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CPUFREQ = "/sys/devices/system/cpu/cpu{}/cpufreq"
PROFILE = Path("/sys/firmware/acpi/platform_profile")
NO_TURBO = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
#: A core busy-looping below this is clamped (PROCHOT runs near 800 MHz).
MIN_LOADED_MHZ = 1500


def read(path: Path) -> str | None:
    """``path``'s stripped text, or ``None`` when the file is absent or unreadable."""
    try:
        return path.read_text().strip()
    except OSError:
        return None


def loaded_mhz(cpu: int) -> float | None:
    """MHz of ``cpu`` after busy-looping on it for one second; the caller's affinity is restored."""
    # The benchmark runner calls this in-process; a mask left behind would pin every rep it launches.
    saved = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cpu})
    try:
        end = time.perf_counter() + 1.0
        while time.perf_counter() < end:
            pass
        khz = read(Path(CPUFREQ.format(cpu), "scaling_cur_freq"))
    finally:
        os.sched_setaffinity(0, saved)
    return int(khz) / 1000 if khz else None


def clock_problems() -> list[str]:
    """Findings about the clock and its clamps."""
    problems = []
    cpu = min(os.sched_getaffinity(0))
    mhz = loaded_mhz(cpu)
    if mhz is not None and mhz < MIN_LOADED_MHZ:
        problems.append(f"clock: cpu{cpu} runs {mhz:.0f} MHz under load (PROCHOT?)")
    if read(PROFILE) == "quiet":
        problems.append("clock: platform_profile is quiet (turbo off, perf capped at 50%)")
    if read(NO_TURBO) == "1":
        problems.append("clock: intel_pstate/no_turbo is 1")
    cap, hw = read(Path(CPUFREQ.format(cpu), "scaling_max_freq")), read(Path(CPUFREQ.format(cpu), "cpuinfo_max_freq"))
    if cap and hw and int(cap) < int(hw):
        problems.append(f"clock: scaling_max_freq {int(cap) // 1000} MHz < hardware max {int(hw) // 1000} MHz")
    return problems


def busiest(n: int = 5) -> list[str]:
    """The ``n`` busiest processes other than this one, as ``pcpu pid command``."""
    out = subprocess.run(
        ["ps", "--sort=-pcpu", "-eo", "pcpu=,pid=,args="], check=False, capture_output=True, text=True
    ).stdout
    rows = []
    for line in out.splitlines():
        pcpu, pid, args = line.split(None, 2)
        if int(pid) != os.getpid() and float(pcpu) >= 5:
            rows.append(f"{pcpu:>5}% {pid:>7} {args[:100]}")
    return rows[:n]


def load_problems(max_load: float) -> list[str]:
    """Findings about the 1-minute load average."""
    load1 = os.getloadavg()[0]
    if load1 <= max_load:
        return []
    return [f"load: 1-min load {load1:.1f} > {max_load}; busiest:", *(f"  {row}" for row in busiest())]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Is this host quiet enough to time code on?")
    parser.add_argument("--max-load", type=float, default=1.5, help="highest acceptable 1-min load (default 1.5)")
    parser.add_argument("--wait", type=float, default=0, metavar="SECONDS", help="poll this long for load to settle")
    args = parser.parse_args(argv)

    clock = clock_problems()
    deadline = time.monotonic() + args.wait
    load = load_problems(args.max_load)
    while load and not clock and time.monotonic() < deadline:
        time.sleep(15)
        load = load_problems(args.max_load)
    for line in clock + load:
        print(line)
    if clock or load:
        return 1
    print(f"quiet: clock unclamped, 1-min load {os.getloadavg()[0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
