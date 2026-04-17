"""Iterative sparse REML wrapper around the ace_iter_reml C++ binary.

Drop-in alternative to fit_ace.sparse_reml.fit_sparse_reml when the
direct sparse Cholesky factor is intractable (typically n_fit ≳ 100k or
ill-conditioned V).  Two-phase fit:

  Phase 1  RHE-mc moment-method warm-start (Pazokitoroudi 2020)
  Phase 2  PCG-AI-REML with Jacobi or two-stage deflation preconditioner

The IterREMLResult dataclass mirrors SparseREMLResult so downstream
consumers (atlas plots, gather scripts) treat the two interchangeably.
"""

from .fit import IterREMLResult, default_binary, fit_iter_reml

__all__ = ["IterREMLResult", "default_binary", "fit_iter_reml"]
