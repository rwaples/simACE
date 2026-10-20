"""``simace gather`` copies each verdict into ``verdicts.tsv`` and never decides one (ADR 0028 §6)."""

from __future__ import annotations

import polars as pl
import yaml

from simace.analysis.gather import VERDICT_COLUMNS, gather_verdicts
from simace.analysis.validate.verdict import decide, folder_family, scenario_family, write_verdict
from tests.analysis.test_validate_verdict import TESTS, _reps

BUILD = {"simace_version": "0.1.0", "source": "v0.1.0-1-gabc1234"}


def _read(path) -> list[dict]:
    return pl.read_csv(path, separator="\t", null_values=["NA", ""], infer_schema_length=None).to_dicts()


def test_one_row_for_the_folder_then_one_per_scenario_copied_from_their_files(tmp_path) -> None:
    shifted = _reps(20, seed=1)
    for section in shifted:
        section["cells"]["trait1_FS"]["z"] += 2.0
    by_scenario = {"a": {"onset_tau": _reps(20)}, "b": {"onset_tau": shifted}}
    paths = {name: tmp_path / name / "validation_summary.yaml" for name in ("a", "b", "c")}
    for name, sections in by_scenario.items():
        header = {**BUILD, "family": "scenario", "scenario": name, "seed": 1, "reps": "1-20"}
        write_verdict(paths[name], header, decide(scenario_family(sections, TESTS)))
    folder_path = tmp_path / "validation_summary.yaml"
    folder_header = {**BUILD, "family": "folder", "folder": "f"}
    folder_header["reps_by_scenario"] = {"a": {"seed": 1, "reps": "1-20"}, "b": {"seed": 100, "reps": "1-20"}}
    write_verdict(folder_path, folder_header, decide(folder_family(by_scenario, TESTS)))

    gather_verdicts("f", folder_path, paths, tmp_path / "verdicts.tsv")

    rows = _read(tmp_path / "verdicts.tsv")
    assert list(rows[0]) == list(VERDICT_COLUMNS)
    assert [(r["family"], r["name"], r["verdict"]) for r in rows] == [
        ("folder", "f", "fail"),
        ("scenario", "a", "pass"),
        ("scenario", "b", "fail"),
        ("scenario", "c", None),
    ]
    folder, a, b, c = rows
    assert folder["n_tests"] == 2 * 27 + 26
    assert folder["reps"] == "a 1-20; b 1-20"
    assert "b/onset_tau/trait1_FS/mean_z" in folder["failed"].split("; ")
    assert "pooled/onset_tau/trait1_FS/mean_z" in folder["failed"].split("; ")
    assert (a["reps"], a["n_tests"], a["failed"], a["source"]) == ("1-20", 27, None, BUILD["source"])
    assert b["failed"].split("; ")[0] == "onset_tau/trait1_FS/mean_z"
    assert all(c[column] is None for column in VERDICT_COLUMNS if column not in {"family", "name"})


def test_insufficient_tests_carry_their_reasons(tmp_path) -> None:
    path = tmp_path / "a" / "validation_summary.yaml"
    write_verdict(
        path, {**BUILD, "family": "scenario", "reps": "1-5"}, decide(scenario_family({"onset_tau": _reps(5)}, TESTS))
    )
    gather_verdicts("f", tmp_path / "validation_summary.yaml", {"a": path}, tmp_path / "verdicts.tsv")
    folder, a = _read(tmp_path / "verdicts.tsv")
    assert folder["verdict"] is None
    assert (a["verdict"], a["n_tests"], a["alpha_each"]) == ("insufficient", 0, None)
    assert a["insufficient"] == "onset_tau: 5 rep(s); the verdict needs 20"


def test_a_file_without_the_schema_is_not_copied(tmp_path, caplog) -> None:
    path = tmp_path / "a" / "validation_summary.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump({"verdict": "pass", "n_tests": 3}))
    with caplog.at_level("WARNING"):
        gather_verdicts("f", tmp_path / "validation_summary.yaml", {"a": path}, tmp_path / "verdicts.tsv")
    assert _read(tmp_path / "verdicts.tsv")[1]["verdict"] is None
    assert "Ignoring 1 validation_summary.yaml file(s)" in caplog.text
