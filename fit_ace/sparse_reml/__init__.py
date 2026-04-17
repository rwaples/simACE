"""Python wrapper around the ace_sreml C++ binary.

Run a sparse REML fit on pedigree-derived GRMs.  The binary lives in
``external/ace_sreml/build/ace_sreml`` after ``cmake --build external/ace_sreml/build``;
its path can be overridden via the ``ACE_SREML_BIN`` environment variable.
"""

from fit_ace.sparse_reml.fit import SparseREMLResult, fit_sparse_reml

__all__ = ["SparseREMLResult", "fit_sparse_reml"]
