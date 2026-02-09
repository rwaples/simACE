"""
Plot phenotype distributions aggregated across replicates for a scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar


def load_phenotypes(parquet_paths):
    """Load and concatenate phenotype parquet files with rep labels."""
    dfs = []
    for i, p in enumerate(parquet_paths):
        df = pd.read_parquet(p)
        df["rep"] = i + 1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_death_age_distribution(df, censor_age, output_path, scenario=""):
    """Plot mortality rate and cumulative mortality by decade."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute mortality rate per decade
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates = []
    decade_labels = []
    for i in range(len(decade_edges) - 1):
        lo, hi = decade_edges[i], decade_edges[i + 1]
        if lo >= censor_age:
            break
        alive_at_start = len(df[df["death_age"] >= lo])
        died_in_decade = len(df[(df["death_age"] >= lo) & (df["death_age"] < hi) & (df["death_age"] < censor_age)])
        rate = died_in_decade / alive_at_start if alive_at_start > 0 else 0
        mortality_rates.append(rate)
        decade_labels.append(f"{int(lo)}-{int(hi - 1)}")

    mortality_rates = np.array(mortality_rates)

    # Left: mortality rate per decade
    axes[0].bar(decade_labels, mortality_rates, edgecolor="black", alpha=0.7)
    axes[0].set_title("Mortality Rate by Decade")
    axes[0].set_xlabel("Age Decade")
    axes[0].set_ylabel("Mortality Rate")
    axes[0].tick_params(axis="x", rotation=45)

    # Right: cumulative mortality per decade with survival annotations
    survival = np.cumprod(1 - mortality_rates)
    cumulative = 1 - survival
    bars = axes[1].bar(decade_labels, cumulative, edgecolor="black", alpha=0.7)
    for bar, s in zip(bars, survival):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"S={s:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_title("Cumulative Mortality by Decade")
    axes[1].set_xlabel("Age Decade")
    axes[1].set_ylabel("Cumulative Mortality")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Death Age Distribution [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_phenotype(df, output_path, scenario=""):
    """Plot phenotype distributions for both traits in a 2x2 grid (trait 1 top, trait 2 bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        death_censored_col = f"death_censored{trait_num}"

        affected = df[df[affected_col] == True]
        death_censored = df[(df[affected_col] == False) & (df[death_censored_col] == True)]

        axes[row, 0].hist(affected[t_col].dropna(), bins=50, density=True, edgecolor="black", alpha=0.7, color="C3")
        axes[row, 0].set_title(f"Trait {trait_num}: Age at Onset (affected)")
        axes[row, 0].set_xlabel("Age")
        axes[row, 0].set_ylabel("Density")

        axes[row, 1].hist(death_censored[t_col].dropna(), bins=50, density=True, edgecolor="black", alpha=0.7, color="C0")
        axes[row, 1].set_title(f"Trait {trait_num}: Age at Death (death-censored, unaffected)")
        axes[row, 1].set_xlabel("Age")
        axes[row, 1].set_ylabel("Density")

    fig.suptitle(f"Phenotype Distributions [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_regression(df, output_path, scenario=""):
    """Plot liability vs age at onset for both traits as jointplots side by side."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Liability vs Age at Onset (affected) [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(1, 2, figure=fig, wspace=0.35)

    for i, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        liability_col = f"liability{trait_num}"

        affected = df[df[affected_col] == True].dropna(subset=[liability_col, t_col])
        x = affected[liability_col].values
        y = affected[t_col].values

        coeffs = np.polyfit(x, y, 1)
        ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[i],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        ax_joint.scatter(x, y, alpha=0.05, s=3, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax_joint.plot(x_line, np.polyval(coeffs, x_line), color="C3", linewidth=2)
        ax_joint.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax_joint.transAxes,
                      va="top", fontsize=12)
        ax_joint.set_xlabel("Liability")
        ax_joint.set_ylabel("Age at Onset")

        ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
        ax_marg_y.hist(y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7)
        ax_marg_x.set_title(f"Trait {trait_num}", fontsize=12)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_joint(df, output_path, scenario=""):
    """2x2 grid of seaborn jointplots: Liability, A, C, E (trait 1 vs trait 2)."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panels = [
        ("liability1", "liability2", "Liability"),
        ("A1", "A2", "A (Additive genetic)"),
        ("C1", "C2", "C (Common environment)"),
        ("E1", "E2", "E (Unique environment)"),
    ]
    panels = [(x, y, t) for x, y, t in panels if x in df.columns and y in df.columns]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Cross-Trait Correlations [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    for idx, (xcol, ycol, title) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        x, y = df[xcol].values, df[ycol].values
        ax_joint.scatter(x, y, alpha=0.05, s=3, rasterized=True)
        ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
        ax_marg_y.hist(y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7)

        r = df[[xcol, ycol]].corr().iloc[0, 1]
        ax_joint.text(0.05, 0.95, f"r = {r:.4f}", transform=ax_joint.transAxes,
                      va="top", fontsize=11)
        ax_joint.set_xlabel(f"{title} (Trait 1)")
        ax_joint.set_ylabel(f"{title} (Trait 2)")

        ax_marg_x.set_title(f"{title}: Trait 1 vs Trait 2", fontsize=11)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        # Hide corner cell
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_violin(df, output_path, scenario=""):
    """Split violin plot of liability by trait, split on affected status."""
    violin_data = pd.concat(
        [
            pd.DataFrame({
                "Trait": "Trait 1",
                "Liability": df["liability1"],
                "Affected": df["affected1"],
            }),
            pd.DataFrame({
                "Trait": "Trait 2",
                "Liability": df["liability2"],
                "Affected": df["affected2"],
            }),
        ],
        ignore_index=True,
    )

    prev1 = df["affected1"].mean()
    prev2 = df["affected2"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=violin_data, x="Trait", y="Liability", hue="Affected",
        split=True, ax=ax,
    )
    ax.set_title(f"Liability Distribution by Trait and Affected Status [{scenario}]")
    ax.text(0, ax.get_ylim()[0], f"Prevalence: {prev1:.1%}",
            ha="center", va="top", fontsize=10, fontstyle="italic")
    ax.text(1, ax.get_ylim()[0], f"Prevalence: {prev2:.1%}",
            ha="center", va="top", fontsize=10, fontstyle="italic")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_incidence(df, censor_age, output_path, scenario=""):
    """Plot cumulative incidence by age for both traits on shared y-axis."""
    ages = np.linspace(0, censor_age, 200)
    n = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for trait_num, ax in zip([1, 2], axes):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"

        cum_inc = np.array([(df[affected_col] & (df[t_col] <= a)).sum() / n for a in ages])
        ax.plot(ages, cum_inc, color="C0", linewidth=2)

        # Find age when 50% of lifetime cases are diagnosed
        lifetime_prev = cum_inc[-1]
        half_target = lifetime_prev / 2
        idx_50 = np.searchsorted(cum_inc, half_target)
        age_50 = ages[min(idx_50, len(ages) - 1)]

        ax.axhline(half_target, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(age_50, color="grey", linestyle="--", linewidth=0.8)
        ax.plot(age_50, half_target, "o", color="C3", markersize=6, zorder=5)
        ax.annotate(
            f"50% of cases\nby age {age_50:.0f}",
            xy=(age_50, half_target), xytext=(12, 12), textcoords="offset points",
            fontsize=10, ha="left", va="bottom",
        )

        ax.set_title(f"Trait {trait_num}")
        ax.set_xlabel("Age")

    axes[0].set_ylabel("Cumulative Incidence")

    fig.suptitle(f"Cumulative Incidence by Age [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_censoring_windows(df, censor_age, young_gen_censoring, middle_gen_censoring,
                           old_gen_censoring, output_path, scenario=""):
    """Plot per-generation censoring windows as cumulative incidence of affected individuals."""
    if "generation" not in df.columns:
        print("Skipping censoring_windows plot: no 'generation' column in data")
        return

    max_gen = df["generation"].max()
    gen_order = [max_gen - 2, max_gen - 1, max_gen]
    gen_labels = [
        f"Old (gen {max_gen - 2})",
        f"Middle (gen {max_gen - 1})",
        f"Young (gen {max_gen})",
    ]
    gen_windows = [old_gen_censoring, middle_gen_censoring, young_gen_censoring]
    traits = [1, 2]
    ages = np.linspace(0, censor_age, 300)

    fig, axes = plt.subplots(len(traits), len(gen_order), figsize=(5 * len(gen_order), 4 * len(traits)),
                             sharex=True, sharey=True, squeeze=False)

    for col, (gen, label, (win_lo, win_hi)) in enumerate(zip(gen_order, gen_labels, gen_windows)):
        gen_df = df[df["generation"] == gen]
        n_gen = len(gen_df)
        if n_gen == 0:
            for row in range(len(traits)):
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center",
                                    transform=axes[row, col].transAxes)
            continue

        for row, trait_num in enumerate(traits):
            ax = axes[row, col]
            t_raw = gen_df[f"t{trait_num}"].values
            t_obs = gen_df[f"t_observed{trait_num}"].values
            affected = gen_df[f"affected{trait_num}"].values

            # Cumulative incidence: raw (all onsets, no censoring)
            raw_inc = np.array([(t_raw <= a).sum() / n_gen for a in ages])
            # Cumulative incidence: observed affected only
            obs_inc = np.array([(affected & (t_obs <= a)).sum() / n_gen for a in ages])

            ax.plot(ages, raw_inc, color="gray", alpha=0.7, linewidth=2, label="True")
            ax.fill_between(ages, raw_inc, alpha=0.15, color="gray")
            ax.plot(ages, obs_inc, color="C0", linewidth=2, label="Observed")
            ax.fill_between(ages, obs_inc, alpha=0.2, color="C0")

            if win_lo > 0:
                ax.axvline(win_lo, color="red", linestyle="--", linewidth=1.5, label="Window")
                ax.axvspan(0, win_lo, alpha=0.08, color="red")
            if win_hi < censor_age:
                ax.axvline(win_hi, color="red", linestyle="--", linewidth=1.5,
                           label="Window" if win_lo <= 0 else None)
                ax.axvspan(win_hi, censor_age, alpha=0.08, color="red")

            pct_affected = affected.mean() * 100
            left_cens = (t_raw < win_lo).sum() / n_gen * 100
            right_cens = (t_raw > win_hi).sum() / n_gen * 100
            death_censored = gen_df[f"death_censored{trait_num}"].values
            pct_death_cens = (death_censored & ~(t_raw < win_lo) & ~(t_raw > win_hi)).mean() * 100
            ax.text(0.03, 0.95,
                    f"Affected: {pct_affected:.1f}%\n"
                    f"Left-cens: {left_cens:.1f}%\n"
                    f"Right-cens: {right_cens:.1f}%\n"
                    f"Death-cens: {pct_death_cens:.1f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nCumulative Incidence")
            if row == len(traits) - 1:
                ax.set_xlabel("Age")

    from matplotlib.lines import Line2D
    legend_ax = axes[0, 1]
    handles, labels = legend_ax.get_legend_handles_labels()
    if "Window" not in labels:
        handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5))
        labels.append("Window")
    legend_ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.suptitle(f"Censoring Windows by Generation [{scenario}]", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def tetrachoric_corr(a, b):
    """Estimate tetrachoric correlation from two binary arrays via MLE.

    Maximizes the bivariate normal log-likelihood over the correlation
    parameter r, given thresholds derived from marginal proportions.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)

    # 2x2 contingency table
    n11 = np.sum(a & b)
    n10 = np.sum(a & ~b)
    n01 = np.sum(~a & b)
    n00 = np.sum(~a & ~b)

    # Marginal proportions
    p_a = a.mean()
    p_b = b.mean()

    # Edge cases: no variation in one or both variables
    if p_a == 0 or p_a == 1 or p_b == 0 or p_b == 1:
        return np.nan

    # Thresholds from marginals
    t_a = norm.ppf(1 - p_a)
    t_b = norm.ppf(1 - p_b)

    def bvn_cdf(x1, x2, r):
        """CDF of standard bivariate normal at (x1, x2) with correlation r."""
        cov = np.array([[1, r], [r, 1]])
        return multivariate_normal.cdf([x1, x2], mean=[0, 0], cov=cov)

    def neg_log_lik(r):
        # P(A=0, B=0) = P(X < t_a, Y < t_b)
        p00 = bvn_cdf(t_a, t_b, r)
        # P(A=0, B=1) = P(X < t_a) - P(X < t_a, Y < t_b)
        p01 = norm.cdf(t_a) - p00
        # P(A=1, B=0) = P(Y < t_b) - P(X < t_a, Y < t_b)
        p10 = norm.cdf(t_b) - p00
        # P(A=1, B=1) = 1 - p00 - p01 - p10
        p11 = 1 - p00 - p01 - p10

        # Clamp to avoid log(0)
        eps = 1e-15
        p11 = max(p11, eps)
        p10 = max(p10, eps)
        p01 = max(p01, eps)
        p00 = max(p00, eps)

        return -(n11 * np.log(p11) + n10 * np.log(p10) +
                 n01 * np.log(p01) + n00 * np.log(p00))

    result = minimize_scalar(neg_log_lik, bounds=(-0.999, 0.999), method="bounded")
    return result.x


def extract_relationship_pairs(df):
    """Extract relationship pairs from phenotype data.

    Returns dict mapping pair type to list of ((rep,id1), (rep,id2)) tuples.
    Pair types: MZ twin, Full sib, Half sib, Parent-offspring, 1st cousin.
    """
    pairs = {
        "MZ twin": [], "Full sib": [], "Half sib": [],
        "Parent-offspring": [], "1st cousin": [],
    }

    # Ensure rep column exists
    has_rep = "rep" in df.columns
    df_work = df if has_rep else df.assign(rep=0)

    # MZ twins: twin != -1, deduplicate by keeping id < twin
    twins = df_work[df_work["twin"] != -1]
    ta = twins["id"].values.astype(int)
    tb = twins["twin"].values.astype(int)
    tr = twins["rep"].values.astype(int)
    mask = ta < tb
    pairs["MZ twin"] = [
        ((r, a), (r, b)) for r, a, b in zip(tr[mask], ta[mask], tb[mask])
    ]

    # Full sibs and half sibs from non-twin offspring
    non_twin_nf = df_work[(df_work["mother"] != -1) & (df_work["twin"] == -1)]
    group_key = ["rep", "mother"]
    sib_counts = non_twin_nf.groupby(group_key).size()
    multi_keys = sib_counts[sib_counts >= 2].index

    multi_sib = non_twin_nf.set_index(group_key).loc[multi_keys].reset_index()

    for _, group in multi_sib.groupby(group_key):
        reps = group["rep"].values
        ids = group["id"].values
        fathers = group["father"].values
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                pi = (reps[i], ids[i])
                pj = (reps[j], ids[j])
                if fathers[i] == fathers[j]:
                    pairs["Full sib"].append((pi, pj))
                else:
                    pairs["Half sib"].append((pi, pj))

    # Parent-offspring: pair each non-founder with their mother and father
    all_nf = df_work[df_work["mother"] != -1]
    reps = all_nf["rep"].values.astype(int)
    child_ids = all_nf["id"].values.astype(int)
    mother_ids = all_nf["mother"].values.astype(int)
    father_ids = all_nf["father"].values.astype(int)
    pairs["Parent-offspring"] = (
        [((r, c), (r, m)) for r, c, m in zip(reps, child_ids, mother_ids)] +
        [((r, c), (r, f)) for r, c, f in zip(reps, child_ids, father_ids)]
    )

    # 1st cousins (maternal line): children whose mothers share the same grandmother
    mother_lookup = df_work[["rep", "id", "mother"]].rename(
        columns={"id": "_mid", "mother": "grandmother"}
    )
    nf_gm = all_nf[["rep", "id", "mother"]].merge(
        mother_lookup, left_on=["rep", "mother"], right_on=["rep", "_mid"], how="left"
    )
    nf_gm = nf_gm[nf_gm["grandmother"].notna() & (nf_gm["grandmother"] != -1)]
    nf_gm["grandmother"] = nf_gm["grandmother"].astype(int)

    for (rep, gm), gm_group in nf_gm.groupby(["rep", "grandmother"]):
        by_mother = gm_group.groupby("mother")["id"].apply(list).to_dict()
        mothers = list(by_mother.values())
        for mi in range(len(mothers)):
            for mj in range(mi + 1, len(mothers)):
                for ci in mothers[mi]:
                    for cj in mothers[mj]:
                        pairs["1st cousin"].append(
                            ((int(rep), int(ci)), (int(rep), int(cj)))
                        )

    return pairs


def plot_tetrachoric_sibling(df, output_path, scenario, A1, C1, A2, C2):
    """Plot tetrachoric correlations by relationship type."""
    pairs = extract_relationship_pairs(df)
    pair_types = ["MZ twin", "Full sib", "Parent-offspring", "Half sib", "1st cousin"]
    bar_colors = ["C0", "C1", "C3", "C2", "C4"]

    # Expected liability-scale correlations per trait
    # Parent-offspring: avg of mother-child (0.5A+C) and father-child (0.5A)
    # 1st cousin (maternal): mothers are sisters, children share C through grandmother
    expected = {
        1: {
            "MZ twin": A1 + C1,
            "Full sib": 0.5 * A1 + C1,
            "Parent-offspring": 0.5 * A1 + 0.5 * C1,
            "Half sib": 0.25 * A1 + C1,
            "1st cousin": 0.125 * A1 + C1,
        },
        2: {
            "MZ twin": A2 + C2,
            "Full sib": 0.5 * A2 + C2,
            "Parent-offspring": 0.5 * A2 + 0.5 * C2,
            "Half sib": 0.25 * A2 + C2,
            "1st cousin": 0.125 * A2 + C2,
        },
    }

    has_rep = "rep" in df.columns
    if has_rep:
        df_idx = df.set_index(["rep", "id"])
    else:
        df_idx = df.assign(rep=0).set_index(["rep", "id"])
    valid_keys = set(df_idx.index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col_idx, trait_num in enumerate([1, 2]):
        ax = axes[col_idx]
        affected_col = f"affected{trait_num}"
        observed_corrs = []
        n_pairs_list = []

        for ptype in pair_types:
            pair_list = pairs[ptype]
            valid = [(a, b) for a, b in pair_list if a in valid_keys and b in valid_keys]
            if len(valid) < 10:
                observed_corrs.append(np.nan)
                n_pairs_list.append(len(valid))
                continue

            ids1, ids2 = zip(*valid)
            a_vals = df_idx.loc[list(ids1), affected_col].values.astype(bool)
            b_vals = df_idx.loc[list(ids2), affected_col].values.astype(bool)

            r_tet = tetrachoric_corr(a_vals, b_vals)
            observed_corrs.append(r_tet)
            n_pairs_list.append(len(valid))

        x = np.arange(len(pair_types))
        bars = ax.bar(x, observed_corrs, width=0.6, color=bar_colors,
                      edgecolor="black", alpha=0.8, zorder=3)

        # Annotate N pairs
        for i, (bar, n_p) in enumerate(zip(bars, n_pairs_list)):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f"N={n_p}", ha="center", va="bottom", fontsize=8)

        # Expected lines
        for i, ptype in enumerate(pair_types):
            exp_val = expected[trait_num][ptype]
            ax.hlines(exp_val, i - 0.35, i + 0.35, colors="black",
                      linestyles="dashed", linewidth=2, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_types, fontsize=9)
        ax.set_ylabel("Tetrachoric Correlation")
        ax.set_title(f"Trait {trait_num}")
        ax.set_ylim(-0.1, 1.1)

        # Legend for expected lines
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color="black", linestyle="--",
                                  linewidth=2, label="Expected")]
        ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(f"Tetrachoric Correlation by Sibling Type [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(parquet_paths, censor_age, output_dir, young_gen_censoring=None,
         middle_gen_censoring=None, old_gen_censoring=None,
         A1=None, C1=None, A2=None, C2=None):
    """Generate all phenotype plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive scenario name from path: results/{scenario}/plots
    scenario = output_dir.parent.name

    sns.set_theme(style="whitegrid")
    df = load_phenotypes(parquet_paths)

    plot_death_age_distribution(df, censor_age, output_dir / "death_age_distribution.png", scenario)
    plot_trait_phenotype(df, output_dir / "phenotype_traits.png", scenario)
    plot_trait_regression(df, output_dir / "liability_regression.png", scenario)
    plot_liability_joint(df, output_dir / "liability_joint.png", scenario)
    plot_liability_violin(df, output_dir / "liability_violin.png", scenario)
    plot_cumulative_incidence(df, censor_age, output_dir / "cumulative_incidence.png", scenario)
    if young_gen_censoring is not None:
        plot_censoring_windows(df, censor_age, young_gen_censoring, middle_gen_censoring,
                               old_gen_censoring, output_dir / "censoring_windows.png", scenario)

    if A1 is not None:
        plot_tetrachoric_sibling(df, output_dir / "tetrachoric_sibling.png", scenario,
                                 A1, C1, A2, C2)

    print(f"Phenotype plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    try:
        parquet_paths = snakemake.input.phenotypes
        censor_age = snakemake.params.censor_age
        young_gen_censoring = snakemake.params.young_gen_censoring
        middle_gen_censoring = snakemake.params.middle_gen_censoring
        old_gen_censoring = snakemake.params.old_gen_censoring
        A1 = snakemake.params.A1
        C1 = snakemake.params.C1
        A2 = snakemake.params.A2
        C2 = snakemake.params.C2
        output_dir = Path(snakemake.output[0]).parent
    except NameError:
        if len(sys.argv) >= 4:
            censor_age = float(sys.argv[1])
            output_dir = sys.argv[2]
            parquet_paths = sys.argv[3:]
            young_gen_censoring = None
            middle_gen_censoring = None
            old_gen_censoring = None
            A1 = None
            C1 = None
            A2 = None
            C2 = None
        else:
            print("Usage: plot_phenotype.py <censor_age> <output_dir> <phenotype1.parquet> ...")
            sys.exit(1)

    main(parquet_paths, censor_age, output_dir, young_gen_censoring,
         middle_gen_censoring, old_gen_censoring, A1, C1, A2, C2)
