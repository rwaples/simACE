"""fp32 vs fp64 parity tests.

The README and wrapper docstring claim σ² estimates match to ~4 decimal
places between the two builds (fit.py:18, README.md:16-19).  These
tests exercise each build individually (smoke) and then assert
parity on the same seed + inputs.
"""

from __future__ import annotations

import pytest

from fit_ace.iter_reml.fit import fit_iter_reml
from tests.fit_ace.iter_reml.conftest import (
    _FP32,
    _FP64,
    _HAS_FP32,
    _HAS_FP64,
    needs_both,
)

_builds: list[tuple[str, object]] = []
if _HAS_FP32:
    _builds.append(("fp32", _FP32))
if _HAS_FP64:
    _builds.append(("fp64", _FP64))


@pytest.mark.skipif(not _builds, reason="no ace_iter_reml build found")
@pytest.mark.parametrize(("label", "binary"), _builds, ids=[b[0] for b in _builds])
class TestEachBuildHealthy:
    def test_smoke(self, tiny_ace_inputs, fast_kwargs, label, binary):
        """Each build individually recovers σ² within ±0.25 on the tiny
        fixture.  Guards against a broken fp32 build sneaking past when
        fp64 works (and vice versa)."""
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            binary=binary,
            **fast_kwargs,
        )
        assert r.converged, f"{label} failed to converge"
        vc = r.vc.set_index("vc_name")["estimate"]
        for name, key in [("V(A)", "var_a"), ("V(C)", "var_c"), ("Ve", "var_e")]:
            est = float(vc[name])
            ref = inp["truth"][key]
            assert est == pytest.approx(ref, abs=0.25), f"{label} {name}: est={est:.4f} truth={ref:.4f}"


@needs_both
class TestFp32Fp64Parity:
    def test_sigma2_agree_to_4dp(self, tiny_ace_inputs, fast_kwargs):
        """σ² estimates from fp32 and fp64 must agree to ~4 decimal
        places on identical inputs + seed (per the claim in
        fit.py:18 and README.md:16).  Uses abs=5e-4, which is one
        sigma in the AI-REML SE band at this fixture size and
        comfortably looser than the claimed 1e-4."""
        inp = tiny_ace_inputs
        kw = {**fast_kwargs, "seed": 42}
        r32 = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            binary=_FP32,
            **kw,
        )
        r64 = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            binary=_FP64,
            **kw,
        )
        v32 = r32.vc.set_index("vc_name")["estimate"]
        v64 = r64.vc.set_index("vc_name")["estimate"]
        for name in ("V(A)", "V(C)", "Ve"):
            assert float(v32[name]) == pytest.approx(float(v64[name]), abs=5e-4), (
                f"fp32/fp64 diverged on {name}: {float(v32[name]):.6f} vs {float(v64[name]):.6f}"
            )
