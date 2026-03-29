"""Shared constants for fit_ace modules."""

# Canonical ordering of EPIMIGHT relationship kinds
KIND_ORDER: list[str] = ["PO", "FS", "HS", "mHS", "pHS", "Av", "1G", "1C"]

# Matplotlib color per kind (C0..C7)
KIND_COLORS: dict[str, str] = {k: f"C{i}" for i, k in enumerate(KIND_ORDER)}

# Human-readable labels
KIND_LABELS: dict[str, str] = {
    "PO": "Parent-Offspring",
    "FS": "Full Sibling",
    "HS": "Half Sibling",
    "mHS": "Maternal HS",
    "pHS": "Paternal HS",
    "Av": "Avuncular",
    "1G": "Grandparent-GC",
    "1C": "1st Cousin",
}
