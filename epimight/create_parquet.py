import pandas as pd
import numpy as np
import json

# TODO: to take with snakemake
PARQUET_PATH = "../results/base/baseline1M/rep1/phenotype.parquet"

OUT_NDD = f"NDD.parquet"
OUT_NDG = f"NDG.parquet"

df = pd.read_parquet(PARQUET_PATH)

# ------------------------------------------------
# mapping id -> row
# ------------------------------------------------
df_indexed = df.set_index("id")

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

out_ndd = build_output("affected1", "t_observed1", diagnosed1)
out_ndg = build_output("affected2", "t_observed2", diagnosed2)

# ------------------------------------------------
# DIAGNOSTICS (unchanged)
# ------------------------------------------------

ybar = "─" * 8

def summarize_tte(tte: pd.DataFrame, name: str, earliest_onset: int = 1):
    print(f"\n=== {name}: check columns & dtypes ===")
    print(tte.dtypes)

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

    rel_dist = (
        tte["relatives"]
        .value_counts()
        .rename_axis("relatives_code")
        .reset_index(name="n")
        .sort_values("relatives_code")
    )
    print(f"\n{ybar} relatives_code distribution ({name})")
    print(rel_dist.to_string(index=False))

    grp = tte.groupby("born_at_year")

    events_c1 = grp["failure_status"].sum().rename("events_c1")
    n_c1 = grp.size().rename("n_c1")

    parent_idx = tte["diagnosed_relatives"] > 0
    grp_parent = tte.loc[parent_idx].groupby("born_at_year") if parent_idx.any() else None

    n_parent = (grp_parent.size() if grp_parent is not None else pd.Series(dtype=int)).rename("n_parent")
    events_parent = (
        grp_parent["failure_status"].sum() if grp_parent is not None else pd.Series(dtype=int)
    ).rename("events_parent")

    def q(s):
        s = s[s > 0]
        return np.quantile(s, [0.1, 0.5, 0.9]) if len(s) > 0 else [np.nan, np.nan, np.nan]

    q_events = grp.apply(
        lambda g: pd.Series(q(g.loc[g["failure_status"] == 1, "failure_time"]),
                            index=["t10", "t50", "t90"])
    )

    summary = (
        pd.concat([
            n_c1, events_c1,
            n_parent, events_parent,
            q_events
        ], axis=1)
        .fillna(0)
        .astype({
            "n_c1": int,
            "events_c1": int,
            "n_parent": int,
            "events_parent": int,
        })
        .reset_index()
        .sort_values("born_at_year")
    )

    summary["h2_ok"] = (
        (summary["events_c1"] > 0) &
        (summary["n_parent"] > 0) &
        (summary["events_parent"] > 0)
    )

    print(f"\n{ybar} Yearly summary ({name})")
    print(summary.to_string(index=False))

    return summary

print("\n###################### PRE-SAVE DIAGNOSTICS ######################")
ndd_summary = summarize_tte(out_ndd, "NDD", earliest_onset=1)
ndg_summary = summarize_tte(out_ndg, "NDG", earliest_onset=1)

years_ndd_ok = set(ndd_summary.loc[ndd_summary["h2_ok"], "born_at_year"].tolist())
years_ndg_ok = set(ndg_summary.loc[ndg_summary["h2_ok"], "born_at_year"].tolist())
years_both_ok = sorted(years_ndd_ok & years_ndg_ok)

print(f"\n{ybar} h²-suitable years (NDD): {sorted(years_ndd_ok)}")
print(f"{ybar} h²-suitable years (NDG): {sorted(years_ndg_ok)}")
print(f"{ybar} Intersection (usable for GC + meta): {years_both_ok}")

print("\n###################### TRUE GENETIC PARAMETERS ######################")

if not {"A1","C1","E1","A2","C2","E2"}.issubset(df.columns):
    print("[WARN] Missing A1/C1/E1/A2/C2/E2 — cannot compute truth.")
else:
    df["L1"] = df["A1"] + df["C1"] + df["E1"]
    df["L2"] = df["A2"] + df["C2"] + df["E2"]

    h2_trait1_true = np.var(df["A1"]) / np.var(df["L1"])
    h2_trait2_true = np.var(df["A2"]) / np.var(df["L2"])

    gc_true = np.corrcoef(df["A1"], df["A2"])[0, 1]
    phen_corr = np.corrcoef(df["L1"], df["L2"])[0, 1]

    print(f"True h² trait1: {h2_trait1_true:.6f}")
    print(f"True h² trait2: {h2_trait2_true:.6f}")
    print(f"True genetic correlation: {gc_true:.6f}")
    print(f"True phenotypic correlation: {phen_corr:.6f}")

    truth = {
        "h2_trait1_true": float(h2_trait1_true),
        "h2_trait2_true": float(h2_trait2_true),
        "genetic_correlation_true": float(gc_true),
        "phenotypic_correlation_true": float(phen_corr),
    }

    with open(f"true_parameters.json", "w") as f:
        json.dump(truth, f, indent=2)

    print("\nSaved truth parameters → true_parameters.json")

# ------------------------------------------------
# Saving TTE outputs
# ------------------------------------------------
out_ndd.to_parquet(OUT_NDD, index=False)
out_ndg.to_parquet(OUT_NDG, index=False)

print("\nCreated files:")
print(OUT_NDD)
print(OUT_NDG)