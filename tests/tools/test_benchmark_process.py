from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from tools.benchmark.process import ProcessSampler, scan_process_group

if TYPE_CHECKING:
    from pathlib import Path


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except ProcessLookupError:
        pass


def test_process_group_sampler_excludes_unrelated_process(tmp_path: Path):
    child_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); time.sleep(5)"
    benchmark = subprocess.Popen(
        [sys.executable, "-c", child_code, "-m", "simace", "simulate"],
        start_new_session=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        time.sleep(0.2)
        processes = scan_process_group(benchmark.pid)
        assert benchmark.pid in {process.pid for process in processes}
        assert unrelated.pid not in {process.pid for process in processes}
        assert len(processes) >= 2

        sampler = ProcessSampler(benchmark.pid, tmp_path / "samples.jsonl", 0.05)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        summary = sampler.summary()
        assert summary["peak_summed_rss_kb"] > 0
        assert summary["max_individual_rss_kb"] > 0
        assert summary["rule_peaks_kb"]["simulate"] > 0
    finally:
        _stop(benchmark)
        _stop(unrelated)
