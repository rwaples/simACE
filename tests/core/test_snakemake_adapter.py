"""Unit tests for the Snakemake script-wrapper adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from simace.core.snakemake_adapter import (
    cli_or_snakemake,
    run_wrapper,
    write_parquet_plain,
)


def _make_snakemake(*, input_attrs, output_attrs, params, wildcards=None, log=None, tmp_path=None):
    """Build a fake ``snakemake`` namespace shaped like the real one."""
    return SimpleNamespace(
        input=SimpleNamespace(**input_attrs),
        output=SimpleNamespace(**output_attrs),
        params=SimpleNamespace(**params),
        wildcards=SimpleNamespace(**(wildcards or {})),
        log=log or [str(tmp_path / "wrapper.log")] if tmp_path else ["/tmp/test.log"],
    )


# ---------------------------------------------------------------------------
# run_wrapper
# ---------------------------------------------------------------------------


class TestRunWrapper:
    def test_introspects_kwargs_from_params(self, tmp_path):
        """Domain function's keyword-only params are pulled from snakemake.params."""

        def domain(df, *, multiplier, offset):
            return df.assign(x=df["x"] * multiplier + offset)

        in_path = tmp_path / "in.parquet"
        out_path = tmp_path / "out.parquet"
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}).to_parquet(in_path)

        sm = _make_snakemake(
            input_attrs={"df": str(in_path)},
            output_attrs={"result": str(out_path)},
            params={"multiplier": 10, "offset": 1},
            tmp_path=tmp_path,
        )
        run_wrapper(
            sm,
            domain,
            inputs={"df": pd.read_parquet},
            output="result",
            writer=write_parquet_plain,
        )
        result = pd.read_parquet(out_path)
        assert result["x"].tolist() == [11.0, 21.0, 31.0]

    def test_pass_through_loader_passes_raw_path(self, tmp_path):
        """A loader of None passes the raw path string (no IO at adapter)."""
        observed: dict = {}

        def domain(in_path, *, scale):
            observed["in_path"] = in_path
            observed["scale"] = scale
            return pd.DataFrame({"x": [scale]})

        sm = _make_snakemake(
            input_attrs={"in_path": "/some/raw/path.parquet"},
            output_attrs={"out": str(tmp_path / "out.parquet")},
            params={"scale": 7},
            tmp_path=tmp_path,
        )
        run_wrapper(
            sm,
            domain,
            inputs={"in_path": None},
            output="out",
            writer=write_parquet_plain,
        )
        assert observed["in_path"] == "/some/raw/path.parquet"
        assert observed["scale"] == 7

    def test_missing_param_raises_attribute_error(self, tmp_path):
        """A keyword-only param absent from snakemake.params raises AttributeError."""

        def domain(df, *, alpha, beta):
            return df

        in_path = tmp_path / "in.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(in_path)
        sm = _make_snakemake(
            input_attrs={"df": str(in_path)},
            output_attrs={"out": str(tmp_path / "out.parquet")},
            params={"alpha": 0.5},  # beta missing
            tmp_path=tmp_path,
        )
        with pytest.raises(AttributeError, match="beta"):
            run_wrapper(
                sm,
                domain,
                inputs={"df": pd.read_parquet},
                output="out",
                writer=write_parquet_plain,
            )

    def test_writer_invoked_with_result_and_path(self, tmp_path):
        """Writer receives (return_value, snakemake.output.<output>)."""
        recorded: list = []

        def writer(value, path):
            recorded.append((value, path))

        def domain(df, *, k):
            return {"got": k}

        in_path = tmp_path / "in.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(in_path)
        sm = _make_snakemake(
            input_attrs={"df": str(in_path)},
            output_attrs={"target": "/expected/output"},
            params={"k": 42},
            tmp_path=tmp_path,
        )
        run_wrapper(
            sm,
            domain,
            inputs={"df": pd.read_parquet},
            output="target",
            writer=writer,
        )
        assert recorded == [({"got": 42}, "/expected/output")]

    def test_input_param_not_pulled_from_params(self, tmp_path):
        """Param names matching `inputs` keys are skipped during introspection."""

        def domain(df, *, k):
            return df.assign(k=k)

        in_path = tmp_path / "in.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(in_path)
        # `df` exists in params too — adapter must prefer the input loader,
        # not pull `df` as a kwarg from snakemake.params.
        sm = _make_snakemake(
            input_attrs={"df": str(in_path)},
            output_attrs={"out": str(tmp_path / "out.parquet")},
            params={"df": "this should be ignored", "k": 99},
            tmp_path=tmp_path,
        )
        run_wrapper(
            sm,
            domain,
            inputs={"df": pd.read_parquet},
            output="out",
            writer=write_parquet_plain,
        )
        result = pd.read_parquet(tmp_path / "out.parquet")
        assert result["k"].tolist() == [99]


# ---------------------------------------------------------------------------
# cli_or_snakemake
# ---------------------------------------------------------------------------


class TestCliOrSnakemake:
    def test_dispatches_to_wrapper_when_snakemake_present(self):
        called: list[str] = []
        cli_fn = lambda: called.append("cli")  # noqa: E731
        wrapper_fn = lambda: called.append("wrapper")  # noqa: E731

        cli_or_snakemake(cli_fn, wrapper_fn, {"snakemake": object(), "__name__": "wrapper_module"})
        assert called == ["wrapper"]

    def test_dispatches_to_cli_when_main(self):
        called: list[str] = []
        cli_fn = lambda: called.append("cli")  # noqa: E731
        wrapper_fn = lambda: called.append("wrapper")  # noqa: E731

        cli_or_snakemake(cli_fn, wrapper_fn, {"__name__": "__main__"})
        assert called == ["cli"]

    def test_no_dispatch_when_imported_module(self):
        """No snakemake, not __main__ — neither runs (module is being imported for tests)."""
        called: list[str] = []
        cli_fn = lambda: called.append("cli")  # noqa: E731
        wrapper_fn = lambda: called.append("wrapper")  # noqa: E731

        cli_or_snakemake(cli_fn, wrapper_fn, {"__name__": "some.module"})
        assert called == []
