#!/usr/bin/env bash
# pedigree-graph 0.8.0 slice 9b: build from a clean checkout, install the
# artifacts into isolated environments, prove the import resolves there, run the
# package's own suite against the installed wheel, and stage a --target site the
# consumer gate (tools/pg08_release_gate.py run --routing <site>) imports from.
#
#   tools/pg08_wheel_gate.sh <work-dir> [ref]      # ref defaults to v0.8
#
# Writes <work-dir>/{clean,dist,build-venv,wheel-venv,sdist-venv,wheel-site} and
# <work-dir>/wheel-gate.json (sha256s, versions, import paths, pytest exit code).
# Never touches a pixi manifest or lock: the build venv is throwaway and the
# version is SETUPTOOLS_SCM_PRETEND_VERSION so an untagged branch yields 0.8.0.
set -euo pipefail

WORK="$(realpath -m "${1:?work dir}")"
REF="${2:-v0.8}"
VERSION="${PG_VERSION:-0.8.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG="$ROOT/external/pedigree-graph"
PY="$PG/.pixi/envs/default/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY (pixi install in $PG first)" >&2; exit 2; }

mkdir -p "$WORK"
CLEAN="$WORK/clean"
if [ -e "$CLEAN" ]; then git -C "$PG" worktree remove --force "$CLEAN"; fi
git -C "$PG" worktree add --detach "$CLEAN" "$REF"
HEAD_SHA="$(git -C "$CLEAN" rev-parse HEAD)"
[ -z "$(git -C "$CLEAN" status --porcelain)" ] || { echo "clean worktree is dirty" >&2; exit 2; }

rm -rf "$WORK/dist" "$WORK/build-venv"
"$PY" -m venv "$WORK/build-venv"
"$WORK/build-venv/bin/pip" install --quiet build
( cd "$CLEAN" && SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" "$WORK/build-venv/bin/python" -m build --outdir "$WORK/dist" )
WHEEL="$(ls "$WORK"/dist/*.whl)"
SDIST="$(ls "$WORK"/dist/*.tar.gz)"
sha256sum "$WHEEL" "$SDIST"

check_install() {  # <venv> <artifact>
  local venv="$1" artifact="$2"
  rm -rf "$venv"
  "$PY" -m venv "$venv"
  "$venv/bin/pip" install --quiet "$artifact"
  "$venv/bin/python" - "$venv" <<'EOF'
import importlib.metadata as m, pathlib, sys
import pedigree_graph as p
venv = sys.argv[1]
file = pathlib.Path(p.__file__).resolve()
assert str(file).startswith(str(pathlib.Path(venv).resolve()) + "/"), file
assert "site-packages" in file.parts, file
assert (file.parent / "py.typed").is_file(), "py.typed missing"
version = m.version("pedigree-graph")
names = [getattr(p, n) for n in p.__all__]
import pedigree_graph.relationships, pedigree_graph.summaries, pedigree_graph.effective_size, pedigree_graph.typing
print("installed", version, file, len(names), "root names")
EOF
}

check_install "$WORK/wheel-venv" "$WHEEL"
check_install "$WORK/sdist-venv" "$SDIST"

# The package's own suite against the installed wheel.  cwd is the clean tree so
# the parity fixtures and data resolve; -P keeps cwd off sys.path and pytest's
# rootdir insertion adds tests/, not the repo root, so pedigree_graph still
# comes from site-packages (asserted).
"$WORK/wheel-venv/bin/pip" install --quiet "pedigree-graph[test]@file://$WHEEL"
( cd "$CLEAN" && "$WORK/wheel-venv/bin/python" -P -c "import pedigree_graph as p; assert '$WORK/wheel-venv/' in p.__file__, p.__file__" )
set +e
( cd "$CLEAN" && "$WORK/wheel-venv/bin/python" -P -m pytest -q -p no:cacheprovider -m "not slow" tests > "$WORK/wheel-pytest.log" 2>&1 )
PYTEST_RC=$?
set -e
tail -n 3 "$WORK/wheel-pytest.log"

rm -rf "$WORK/wheel-site"
"$PY" -m pip install --quiet --no-deps --target "$WORK/wheel-site" "$WHEEL"

"$PY" - "$WORK" "$HEAD_SHA" "$REF" "$VERSION" "$WHEEL" "$SDIST" "$PYTEST_RC" <<'EOF'
import hashlib, json, pathlib, sys
work, sha, ref, version, wheel, sdist, rc = sys.argv[1:]
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record = {
    "ref": ref, "head": sha, "version": version,
    "wheel": {"file": pathlib.Path(wheel).name, "sha256": digest(wheel)},
    "sdist": {"file": pathlib.Path(sdist).name, "sha256": digest(sdist)},
    "wheel_pytest_exit": int(rc),
    "wheel_site": str(pathlib.Path(work) / "wheel-site"),
}
out = pathlib.Path(work) / "wheel-gate.json"
out.write_text(json.dumps(record, indent=2) + "\n")
print(out.read_text())
EOF
exit "$PYTEST_RC"
