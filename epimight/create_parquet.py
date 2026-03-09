import pandas as pd
import numpy as np

# TODO: to take with snakemake
PARQUET_PATH = "../results/base/baseline1M/rep1/phenotype.parquet"

OUTPUT_FOLDER = "../epimight/"
OUT_NDD = f"{OUTPUT_FOLDER}NDD.parquet"
OUT_NDG = f"{OUTPUT_FOLDER}NDG.parquet"

df = pd.read_parquet(PARQUET_PATH)

# ------------------------------------------------
# mapping id -> row
# ------------------------------------------------
df_indexed = df.set_index("id")

# Affected status (booleans) for parents / twin
mother_aff1 = df_indexed["affected1"]
father_aff1 = df_indexed["affected1"]
twin_aff1 = df_indexed["affected1"]

mother_aff2 = df_indexed["affected2"]
father_aff2 = df_indexed["affected2"]
twin_aff2 = df_indexed["affected2"]

# ------------------------------------------------
# encode relatives (bitmask: 1=mother, 2=father, 4=twin)
# ------------------------------------------------
mother_valid = df["mother"] > 0
father_valid = df["father"] > 0
twin_valid = df["twin"] > 0

relatives_code = (
    mother_valid.astype(int)
    + 2 * father_valid.astype(int)
    + 4 * twin_valid.astype(int)
)

# ------------------------------------------------
# diagnosed relatives (NDD)
# ------------------------------------------------
mother_diag1 = df["mother"].map(mother_aff1).fillna(False)
father_diag1 = df["father"].map(father_aff1).fillna(False)
twin_diag1 = df["twin"].map(twin_aff1).fillna(False)

diagnosed1 = (
    mother_diag1.astype(int)
    + father_diag1.astype(int)
    + twin_diag1.astype(int)
)

# ------------------------------------------------
# diagnosed relatives (NDG)
# ------------------------------------------------
mother_diag2 = df["mother"].map(mother_aff2).fillna(False)
father_diag2 = df["father"].map(father_aff2).fillna(False)
twin_diag2 = df["twin"].map(twin_aff2).fillna(False)

diagnosed2 = (
    mother_diag2.astype(int)
    + father_diag2.astype(int)
    + twin_diag2.astype(int)
)

# ------------------------------------------------
# build outputs (general)
# ------------------------------------------------
def build_output(affected, time, diagnosed):
    out = pd.DataFrame({
        "person_id": df["id"],
        "born_at": df["generation"],
        "born_at_year": 1960 + 20 * df["generation"],
        "dead_at_year": 1960 + 20 * df["generation"] + df["death_age"].astype(int),
        "failure_status": df[affected].astype(int),
        "failure_time": df[time].astype(int),
        "relatives": relatives_code.astype(int),
        "diagnosed_relatives": diagnosed.astype(int),
    })
    return out

# Build the two TTE outputs consistent with the file names
out_ndd = build_output("affected1", "t_observed1", diagnosed1)
out_ndg = build_output("affected2", "t_observed2", diagnosed2)

# ------------------------------------------------
# DIAGNOSTICS: expectations before saving
# ------------------------------------------------
def summarize_tte(tte: pd.DataFrame, name: str, earliest_onset: int = 1):
    print(f"\n=== {name}: check columns & dtypes ===")
    print(tte.dtypes)

    # Basic checks
    assert set([
        "person_id", "born_at", "born_at_year", "dead_at_year",
        "failure_status", "failure_time", "relatives", "diagnosed_relatives"
    ]).issubset(tte.columns)

    if not tte["failure_status"].isin([0, 1]).all():
        print(f"[WARN] {name}: failure_status contains non-binary values.")

    if (tte["failure_time"] < 0).any():
        print(f"[WARN] {name}: failure_time contains negative values.")

    if (tte["dead_at_year"] < tte["born_at_year"]).any():
        print(f"[WARN] {name}: dead_at_year < born_at_year (inconsistent).")

    # Distribution of relatives (bitmask)
    rel_dist = (
        tte["relatives"]
        .value_counts()
        .rename_axis("relatives_code")
        .reset_index(name="n")
        .sort_values("relatives_code")
    )
    print(f"\n{ybar} relatives_code distribution ({name})")
    print(rel_dist.to_string(index=False))

    # Summary by birth year
    grp = tte.groupby("born_at_year")

    # Events in general population (c1)
    events_c1 = grp["failure_status"].sum().rename("events_c1")
    n_c1 = grp.size().rename("n_c1")

    # Individuals with at least one diagnosed relative (define c2/c3)
    parent_idx = tte["diagnosed_relatives"] > 0
    grp_parent = tte.loc[parent_idx].groupby("born_at_year") if parent_idx.any() else None

    n_parent = (grp_parent.size() if grp_parent is not None else pd.Series(dtype=int)).rename("n_parent")
    events_parent = (
        grp_parent["failure_status"].sum() if grp_parent is not None else pd.Series(dtype=int)
    ).rename("events_parent")

    # Events above minimal onset
    events_c1_ge = grp.apply(
        lambda g: int(((g["failure_status"] == 1) & (g["failure_time"] >= earliest_onset)).sum())
    ).rename("events_c1_ge_onset")

    events_parent_ge = (
        tte.loc[parent_idx]
        .groupby("born_at_year")
        .apply(lambda g: int(((g["failure_status"] == 1) & (g["failure_time"] >= earliest_onset)).sum()))
        if parent_idx.any() else pd.Series(dtype=int)
    ).rename("events_parent_ge_onset")

    # Quantiles for failure_time among events
    def q(s):
        s = s[s > 0]
        return np.quantile(s, [0.1, 0.5, 0.9]) if len(s) > 0 else [np.nan, np.nan, np.nan]

    q_events = grp.apply(
        lambda g: pd.Series(q(g.loc[g["failure_status"] == 1, "failure_time"]),
                            index=["t10", "t50", "t90"])
    )

    summary = (
        pd.concat([
            n_c1, events_c1, events_c1_ge,
            n_parent, events_parent, events_parent_ge,
            q_events
        ], axis=1)
        .fillna(0)
        .astype({
            "n_c1": int,
            "events_c1": int,
            "events_c1_ge_onset": int,
            "n_parent": int,
            "events_parent": int,
            "events_parent_ge_onset": int
        })
        .reset_index()
        .sort_values("born_at_year")
    )

    # Flags for h² readiness (minimal requirements for epimight)
    summary["h2_ok"] = (
        (summary["events_c1_ge_onset"] > 0) &
        (summary["n_parent"] > 0) &
        (summary["events_parent_ge_onset"] > 0)
    )

    print(f"\n{ybar} Yearly summary ({name})")
    print(summary.to_string(index=False))

    # Warnings
    bad_years = summary.loc[~summary["h2_ok"], "born_at_year"].tolist()
    if bad_years:
        print(
            f"[WARN] {name}: years NOT suitable for h² "
            f"(missing at least one among: events in c1, parent-of-case, events in parent cohort): {bad_years}"
        )
    else:
        print(f"[OK] {name}: all years suitable for h² with earliest_onset={earliest_onset}")

    return summary


# nice separator
ybar = "─" * 8

print("\n###################### PRE-SAVE DIAGNOSTICS ######################")
ndd_summary = summarize_tte(out_ndd, "NDD", earliest_onset=1)
ndg_summary = summarize_tte(out_ndg, "NDG", earliest_onset=1)

# Intersection of h2-ok years (needed for GC/meta)
years_ndd_ok = set(ndd_summary.loc[ndd_summary["h2_ok"], "born_at_year"].tolist())
years_ndg_ok = set(ndg_summary.loc[ndg_summary["h2_ok"], "born_at_year"].tolist())
years_both_ok = sorted(years_ndd_ok & years_ndg_ok)

print(f"\n{ybar} h²-suitable years (NDD): {sorted(years_ndd_ok)}")
print(f"{ybar} h²-suitable years (NDG): {sorted(years_ndg_ok)}")
print(f"{ybar} Intersection (usable for GC + meta): {years_both_ok}")

if len(years_both_ok) < 2:
    print("[WARN] Warning: fewer than 2 overlapping years → meta-analysis may fail or be unstable.")

# (optional) preview of problematic rows
def preview_problem_years(tte, name, bad_years):
    if not bad_years:
        return
    print(f"\n{ybar} Example rows for problematic years ({name}):")
    print(tte[tte["born_at_year"].isin(bad_years)].head(10).to_string(index=False))

preview_problem_years(out_ndd, "NDD", sorted(set(ndd_summary.loc[~ndd_summary["h2_ok"], "born_at_year"].tolist())))
preview_problem_years(out_ndg, "NDG", sorted(set(ndg_summary.loc[~ndg_summary["h2_ok"], "born_at_year"].tolist())))

# ------------------------------------------------
# Saving
# ------------------------------------------------
out_ndd.to_parquet(OUT_NDD, index=False)
out_ndg.to_parquet(OUT_NDG, index=False)

print("\nCreated files:")
print(OUT_NDD)
print(OUT_NDG)
