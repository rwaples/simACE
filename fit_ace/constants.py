"""Shared constants for fit_ace modules."""

# Canonical ordering of EPIMIGHT relationship kinds
KIND_ORDER: list[str] = ["PO", "FS", "HS", "mHS", "pHS", "Av", "1G", "1C"]

# Muted palette per kind (harmonized with sim_ace PAIR_COLORS)
KIND_COLORS: dict[str, str] = {
    "PO": "#228833",  # muted green (matches MO)
    "FS": "#EE6677",  # muted rose
    "HS": "#66CCEE",  # muted cyan
    "mHS": "#66CCEE",  # muted cyan (same as HS)
    "pHS": "#AA3377",  # muted purple
    "Av": "#999933",  # muted dark olive
    "1G": "#CC6633",  # muted brown
    "1C": "#BBBBBB",  # neutral grey
}

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
