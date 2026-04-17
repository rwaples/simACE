"""Unit tests for ``fit_ace.iter_reml.fit.default_binary``.

Resolution order: ``ACE_ITER_REML_BIN`` env > fp32 build > fp64 build.
"""

from __future__ import annotations

import fit_ace.iter_reml.fit as fit_mod
from fit_ace.iter_reml.fit import default_binary


class TestDefaultBinary:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        monkeypatch.setenv("ACE_ITER_REML_BIN", str(fake))
        assert default_binary() == fake

    def test_env_override_without_existing_path(self, monkeypatch, tmp_path):
        """Env var wins even when the pointed-to file doesn't exist — the
        env override is an explicit user override and should short-circuit
        build discovery regardless of existence (the caller's
        FileNotFoundError handler surfaces the problem downstream)."""
        fake = tmp_path / "does_not_exist"
        monkeypatch.setenv("ACE_ITER_REML_BIN", str(fake))
        assert default_binary() == fake

    def test_fp32_preferred_when_both_exist(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ACE_ITER_REML_BIN", raising=False)
        fp32 = tmp_path / "fp32" / "ace_iter_reml"
        fp64 = tmp_path / "fp64" / "ace_iter_reml"
        fp32.parent.mkdir()
        fp64.parent.mkdir()
        fp32.touch()
        fp64.touch()
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP32", fp32)
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP64", fp64)
        assert default_binary() == fp32

    def test_fp64_fallback_when_fp32_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ACE_ITER_REML_BIN", raising=False)
        fp32 = tmp_path / "fp32" / "ace_iter_reml"  # not created
        fp64 = tmp_path / "fp64" / "ace_iter_reml"
        fp64.parent.mkdir()
        fp64.touch()
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP32", fp32)
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP64", fp64)
        assert default_binary() == fp64

    def test_fp64_returned_when_neither_exists(self, monkeypatch, tmp_path):
        """When neither build is on disk, default_binary() still returns
        the fp64 path (the caller raises FileNotFoundError on missing
        binary with an informative build command)."""
        monkeypatch.delenv("ACE_ITER_REML_BIN", raising=False)
        fp32 = tmp_path / "fp32" / "ace_iter_reml"
        fp64 = tmp_path / "fp64" / "ace_iter_reml"
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP32", fp32)
        monkeypatch.setattr(fit_mod, "_DEFAULT_BINARY_FP64", fp64)
        result = default_binary()
        assert result == fp64
        assert not result.exists()

    def test_default_returns_ace_iter_reml_named_path(self, monkeypatch):
        """Shape check — the resolved default binary is always named
        ``ace_iter_reml`` regardless of build variant."""
        monkeypatch.delenv("ACE_ITER_REML_BIN", raising=False)
        assert default_binary().name == "ace_iter_reml"
