"""Benchmark-only case construction and evaluation helpers.

Nothing in this package may be imported by production replay modules.
"""

from .cases import BatchCaseSampler, compile_sparse_constraints

__all__ = ["BatchCaseSampler", "compile_sparse_constraints"]
