"""Layout of the per-individual component matrix and the preallocated pedigree arrays a simulation fills."""

import numpy as np
import polars as pl

# Columns of the (n, 6) per-individual component matrix ``pheno``.
_A1, _C1, _E1, _A2, _C2, _E2 = range(6)


_TRAIT_COLUMNS = {1: (_A1, _C1, _E1), 2: (_A2, _C2, _E2)}


def _liability(pheno: np.ndarray, trait: int, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    """Return ``A + C + E`` for ``trait`` over ``rows`` of ``pheno``."""
    a, c, e = _TRAIT_COLUMNS[trait]
    return pheno[rows, a] + pheno[rows, c] + pheno[rows, e]


_PED_COLUMNS = [
    "id",
    "sex",
    "mother",
    "father",
    "twin",
    "generation",
    "household_id",
    "A1",
    "C1",
    "E1",
    "liability1",
    "A2",
    "C2",
    "E2",
    "liability2",
]


def _ped_columns(arrays: dict[str, np.ndarray]) -> list[str]:
    """The pedigree's columns: :data:`_PED_COLUMNS`, with ``F`` after ``household_id`` when tracked (ADR 0033)."""
    if "F" not in arrays:
        return _PED_COLUMNS
    at = _PED_COLUMNS.index("household_id") + 1
    return [*_PED_COLUMNS[:at], "F", *_PED_COLUMNS[at:]]


def _init_pedigree_arrays(total_rows: int, inbreeding: bool = False) -> dict[str, np.ndarray]:
    """Pre-allocate numpy arrays for the pedigree.

    Dtype choices for memory efficiency at large N:
    - int32 for IDs (supports up to 2.1B individuals)
    - int8 for sex (0/1)
    - float32 for ACE variance components (~7 significant digits)
    - float64 for liabilities (full precision for phenotype models)
    - float32 for the inbreeding coefficient ``F``, allocated only when
      ``inbreeding`` is set
    """
    _dtypes = {
        "id": np.int32,
        "mother": np.int32,
        "father": np.int32,
        "twin": np.int32,
        "household_id": np.int32,
        "sex": np.int8,
        "generation": np.int32,
        "A1": np.float32,
        "C1": np.float32,
        "E1": np.float32,
        "A2": np.float32,
        "C2": np.float32,
        "E2": np.float32,
        "liability1": np.float64,
        "liability2": np.float64,
    }
    arrays: dict[str, np.ndarray] = {}
    if inbreeding:
        _dtypes["F"] = np.float32
    for col, dtype in _dtypes.items():
        arrays[col] = np.empty(total_rows, dtype=dtype)
    arrays["twin"][:] = -1
    return arrays


def _fill_pedigree_slice(
    arrays: dict[str, np.ndarray],
    offset: int,
    pheno: np.ndarray,
    sex: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    generation: int,
    is_founder: bool,
    id_offset: int,
    parent_offset: int,
    household_offset: int,
    F: np.ndarray | None = None,
) -> None:
    """Fill a slice of pre-allocated pedigree arrays for one generation (``F`` into the ``F`` array when given)."""
    n = len(pheno)
    s = slice(offset, offset + n)

    arrays["id"][s] = np.arange(n, dtype=np.int32) + np.int32(id_offset)
    arrays["sex"][s] = sex
    arrays["generation"][s] = generation
    arrays["household_id"][s] = (household_ids + household_offset).astype(np.int32)
    if F is not None:
        arrays["F"][s] = F

    if is_founder:
        arrays["mother"][s] = -1
        arrays["father"][s] = -1
    else:
        arrays["mother"][s] = (parents[:, 0] + parent_offset).astype(np.int32)
        arrays["father"][s] = (parents[:, 1] + parent_offset).astype(np.int32)

    # Twin column
    arrays["twin"][offset : offset + n] = -1
    if len(twins) > 0:
        twin_ids = (twins + np.int32(id_offset)).astype(np.int32)
        arrays["twin"][offset + twins[:, 0]] = twin_ids[:, 1]
        arrays["twin"][offset + twins[:, 1]] = twin_ids[:, 0]

    # ACE components and liabilities
    for trait, (a, c, e) in _TRAIT_COLUMNS.items():
        arrays[f"A{trait}"][s] = pheno[:, a]
        arrays[f"C{trait}"][s] = pheno[:, c]
        arrays[f"E{trait}"][s] = pheno[:, e]
        arrays[f"liability{trait}"][s] = _liability(pheno, trait)


def _arrays_to_dataframe(arrays: dict[str, np.ndarray], total_rows: int) -> pl.DataFrame:
    """Convert pre-allocated arrays to a pedigree DataFrame."""
    return pl.DataFrame({col: arrays[col][:total_rows] for col in _ped_columns(arrays)})
