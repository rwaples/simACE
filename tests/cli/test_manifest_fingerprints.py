"""Output fingerprints survive a moved mtime when the content is unchanged (issue #80)."""

import os

from simace.cli.manifest import changed_outputs, fingerprints, unchanged


def _bump_mtime(path, ns=10**9):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + ns))


def test_a_fingerprint_records_size_mtime_and_a_content_digest(tmp_path):
    path = tmp_path / "out.yaml"
    path.write_text("a: 1\n")
    fp = fingerprints([path])["out.yaml"]
    assert set(fp) == {"size", "mtime_ns", "blake2b"}
    assert fp["size"] == 5
    assert len(fp["blake2b"]) == 32


def test_a_moved_mtime_with_the_same_content_is_unchanged(tmp_path):
    """What an NFS server does when the writing client returns its write delegation."""
    path = tmp_path / "out.parquet"
    path.write_bytes(b"x" * 4096)
    recorded = fingerprints([path])
    _bump_mtime(path)
    assert unchanged(recorded["out.parquet"], path)
    assert changed_outputs(recorded, [path]) == ()


def test_same_size_new_content_with_a_moved_mtime_is_changed(tmp_path):
    path = tmp_path / "out.parquet"
    path.write_bytes(b"x" * 4096)
    recorded = fingerprints([path])
    path.write_bytes(b"y" * 4096)
    _bump_mtime(path)
    assert changed_outputs(recorded, [path]) == ("out.parquet",)


def test_a_size_change_is_changed_without_reading_the_file(tmp_path):
    path = tmp_path / "out.yaml"
    path.write_text("a: 1\n")
    recorded = fingerprints([path])
    path.write_text("a: 12\n")
    assert changed_outputs(recorded, [path]) == ("out.yaml",)


def test_a_fingerprint_from_before_the_digest_treats_a_moved_mtime_as_changed(tmp_path):
    path = tmp_path / "out.yaml"
    path.write_text("a: 1\n")
    st = path.stat()
    legacy = {"out.yaml": {"size": st.st_size, "mtime_ns": st.st_mtime_ns}}
    assert changed_outputs(legacy, [path]) == ()
    _bump_mtime(path)
    assert changed_outputs(legacy, [path]) == ("out.yaml",)


def test_a_missing_file_or_record_is_changed(tmp_path):
    path = tmp_path / "out.yaml"
    path.write_text("a: 1\n")
    recorded = fingerprints([path])
    assert changed_outputs({}, [path]) == ("out.yaml",)
    path.unlink()
    assert not unchanged(recorded["out.yaml"], path)
