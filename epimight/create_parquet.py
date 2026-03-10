import argparse
import os
import pandas as pd
import numpy as np
import json

from sim_ace.pedigree_graph import extract_relationship_pairs

# Mapping from EPIMIGHT relationship kinds to ACE pair type names
KIND_TO_PAIRS = {
    "PO":  ["Mother-offspring", "Father-offspring"],
    "FS":  ["Full sib", "MZ twin"],
    "HS":  ["Maternal half sib", "Paternal half sib"],
    "mHS": ["Maternal half sib"],
    "pHS": ["Paternal half sib"],
    "1C":  ["1st cousin"],
    "Av":  ["Avuncular"],
    "1G":  ["Grandparent-grandchild"],
}


def count_affected_relatives(pair_list, affected, n):
    """Count how many affected relatives each person has via the given pairs.

    Args:
        pair_list: list of (idx1, idx2) arrays from extract_relationship_pairs
        affected: boolean array of affected status per row index
        n: total number of individuals
    Returns:
        Integer array of length n with counts of affected relatives.
    """
    counts = np.zeros(n, dtype=int)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, affected[idx2].astype(int))
        np.add.at(counts, idx2, affected[idx1].astype(int))
    return counts


def count_total_relatives(pair_list, n):
    """Count total relatives of a given type per person (regardless of affected status)."""
    counts = np.zeros(n, dtype=int)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, 1)
        np.add.at(counts, idx2, 1)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert ACE phenotype parquet to EPIMIGHT TTE format"
    )
    parser.add_argument(
        "--phenotype", required=True,
        help="Path to phenotype.parquet from ACE pipeline"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for NDD.parquet, NDG.parquet, true_parameters.json"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_ndd_path = os.path.join(args.output_dir, "NDD.parquet")
    out_ndg_path = os.path.join(args.output_dir, "NDG.parquet")
    truth_path = os.path.join(args.output_dir, "true_parameters.json")

    df = pd.read_parquet(args.phenotype)
    n = len(df)

    # ------------------------------------------------
    # Extract all relationship pairs via PedigreeGraph
    # ------------------------------------------------
    print("Extracting relationship pairs...")
    all_pairs = extract_relationship_pairs(df)

    # Affected status arrays (by row index)
    affected1 = df["affected1"].values.astype(bool)
    affected2 = df["affected2"].values.astype(bool)

    # ------------------------------------------------
    # Build diagnosed_relatives and n_relatives per kind
    # ------------------------------------------------
    diag_cols_1 = {}
    diag_cols_2 = {}
    nrel_cols = {}

    for kind, pair_types in KIND_TO_PAIRS.items():
        pair_list = [(all_pairs[pt][0], all_pairs[pt][1]) for pt in pair_types if pt in all_pairs]
        diag_cols_1[kind] = count_affected_relatives(pair_list, affected1, n)
        diag_cols_2[kind] = count_affected_relatives(pair_list, affected2, n)
        nrel_cols[kind] = count_total_relatives(pair_list, n)

    # ------------------------------------------------
    # build outputs
    # ------------------------------------------------
    def build_output(affected_col, time_col, diag_cols):
        out = pd.DataFrame({
            "person_id": df["id"],
            "born_at": df["generation"],
            "born_at_year": 1960 + df["generation"],
            "dead_at_year": 1960 + df["generation"] + df["death_age"].astype(int),
            "failure_status": df[affected_col].astype(int),
            "failure_time": df[time_col].astype(int),
        })
        # Per-kind diagnosed relative counts
        for kind in KIND_TO_PAIRS:
            out[f"diagnosed_relatives_{kind}"] = diag_cols[kind].astype(int)
        # Per-kind total relative counts (diagnostics)
        for kind in KIND_TO_PAIRS:
            out[f"n_relatives_{kind}"] = nrel_cols[kind].astype(int)
        return out

    out_ndd = build_output("affected1", "t_observed1", diag_cols_1)
    out_ndg = build_output("affected2", "t_observed2", diag_cols_2)

    # ------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------

    ybar = "─" * 8

    def summarize_tte(tte: pd.DataFrame, name: str):
        print(f"\n=== {name}: check columns & dtypes ===")
        print(tte.dtypes)

        if not tte["failure_status"].isin([0, 1]).all():
            print(f"[WARN] {name}: failure_status contains non-binary values.")

        if (tte["failure_time"] < 0).any():
            print(f"[WARN] {name}: failure_time contains negative values.")

        if (tte["dead_at_year"] < tte["born_at_year"]).any():
            print(f"[WARN] {name}: dead_at_year < born_at_year (inconsistent).")

        # Per-kind summary
        print(f"\n{ybar} Relative counts by kind ({name})")
        for kind in KIND_TO_PAIRS:
            diag_col = f"diagnosed_relatives_{kind}"
            nrel_col = f"n_relatives_{kind}"
            n_with_rel = (tte[nrel_col] > 0).sum()
            n_with_diag = (tte[diag_col] > 0).sum()
            mean_diag = tte.loc[tte[nrel_col] > 0, diag_col].mean() if n_with_rel > 0 else 0
            print(f"  {kind:>3s}: {n_with_rel:>8d} with relatives, "
                  f"{n_with_diag:>8d} with diagnosed relatives, "
                  f"mean diagnosed (among those with rel): {mean_diag:.3f}")

        # Yearly summary using PO as default for the h2 feasibility check
        default_diag = "diagnosed_relatives_PO"
        grp = tte.groupby("born_at_year")

        events_c1 = grp["failure_status"].sum().rename("events_c1")
        n_c1 = grp.size().rename("n_c1")

        parent_idx = tte[default_diag] > 0
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

        print(f"\n{ybar} Yearly summary ({name}, using PO for h2 feasibility)")
        print(summary.to_string(index=False))

        return summary

    print("\n###################### PRE-SAVE DIAGNOSTICS ######################")
    ndd_summary = summarize_tte(out_ndd, "NDD")
    ndg_summary = summarize_tte(out_ndg, "NDG")

    years_ndd_ok = set(ndd_summary.loc[ndd_summary["h2_ok"], "born_at_year"].tolist())
    years_ndg_ok = set(ndg_summary.loc[ndg_summary["h2_ok"], "born_at_year"].tolist())
    years_both_ok = sorted(years_ndd_ok & years_ndg_ok)

    print(f"\n{ybar} h²-suitable years (NDD): {sorted(years_ndd_ok)}")
    print(f"{ybar} h²-suitable years (NDG): {sorted(years_ndg_ok)}")
    print(f"{ybar} Intersection (usable for GC + meta): {years_both_ok}")

    # ------------------------------------------------
    # Pair counts summary
    # ------------------------------------------------
    print(f"\n{ybar} Pair counts from PedigreeGraph")
    for pt_name, (p1, p2) in all_pairs.items():
        print(f"  {pt_name}: {len(p1)} pairs")

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

        with open(truth_path, "w") as f:
            json.dump(truth, f, indent=2)

        print(f"\nSaved truth parameters → {truth_path}")

    # ------------------------------------------------
    # Saving TTE outputs
    # ------------------------------------------------
    out_ndd.to_parquet(out_ndd_path, index=False)
    out_ndg.to_parquet(out_ndg_path, index=False)

    print(f"\nCreated files:")
    print(out_ndd_path)
    print(out_ndg_path)


if __name__ == "__main__":
    main()
