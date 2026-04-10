"""Bivariate PA-FGRS scoring - Snakemake wrapper."""

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    import logging
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from fit_ace.pafgrs.pafgrs import (
        build_kinship_from_pairs,
        compute_empirical_cip,
        compute_thresholds_and_w,
        compute_true_cip_weibull,
    )
    from fit_ace.pafgrs.pafgrs_bivariate import (
        build_pheno_lookups,
        estimate_rg,
        prepare_bivariate_scoring,
        score_bivariate_variant,
    )
    from fit_ace.pafgrs.pafgrs_metrics import compute_bivariate_metrics
    from sim_ace.analysis.ltm_falconer import compute_ltm_falconer
    from sim_ace.core.utils import save_parquet

    logger = logging.getLogger(__name__)
    p = snakemake.params

    pedigree_df = pd.read_parquet(
        snakemake.input.pedigree,
        columns=["id", "mother", "father", "twin", "sex", "generation"],
    )
    phenotype_df = pd.read_parquet(
        snakemake.input.phenotype,
        columns=["id", "generation", "affected1", "affected2", "A1", "A2", "t_observed1", "t_observed2"],
    )

    base_dir = str(Path(snakemake.output.done).parent)

    ndegree = int(p.ndegree)
    censor_age = float(p.censor_age)

    # Build kinship once
    kmat = build_kinship_from_pairs(pedigree_df, ndegree=ndegree)

    # --- Prep once: extract relatives, pre-extract kinship ---
    prep = prepare_bivariate_scoring(pedigree_df, phenotype_df, ndegree, kmat)

    # --- Per-trait CIP tables and h2 estimation ---
    cip_tables = {}
    h2_values = {}

    for trait_num in [1, 2]:
        h2_true = float(getattr(p, f"A{trait_num}"))
        prevalence = float(getattr(p, f"prevalence{trait_num}"))
        beta = float(getattr(p, f"beta{trait_num}"))
        model = getattr(p, f"phenotype_model{trait_num}")
        pheno_params = getattr(p, f"phenotype_params{trait_num}") or {}

        tables = {}
        cip_ages_emp, cip_vals_emp, prev_emp = compute_empirical_cip(
            phenotype_df,
            trait_num,
            max_age=censor_age,
        )
        tables["empirical"] = (cip_ages_emp, cip_vals_emp, prev_emp)

        if model == "frailty" and pheno_params.get("distribution") == "weibull":
            cip_ages_true, cip_vals_true = compute_true_cip_weibull(
                scale=pheno_params["scale"],
                rho=pheno_params["rho"],
                beta=beta,
                max_age=censor_age,
            )
            tables["true"] = (cip_ages_true, cip_vals_true, prevalence)
        else:
            tables["true"] = tables["empirical"]

        cip_tables[trait_num] = tables

        try:
            falconer = compute_ltm_falconer(
                phenotype_df,
                kinds=["FS"],
                trait_num=trait_num,
                seed=int(p.seed),
                pedigree=pedigree_df,
            )
            h2_est = falconer.get("FS", {}).get("h2_falconer")
            if h2_est is None or np.isnan(h2_est):
                h2_est = h2_true
            else:
                h2_est = float(np.clip(h2_est, 0.01, 0.99))
        except Exception:
            logger.warning("Falconer h2 estimation failed for trait %d", trait_num, exc_info=True)
            h2_est = h2_true

        h2_values[trait_num] = {"true": h2_true, "estimated": h2_est}

    # --- rA estimation ---
    rA_true = float(p.rA)
    try:
        rA_estimated = estimate_rg(
            phenotype_df,
            pedigree_df,
            h2_values[1]["estimated"],
            h2_values[2]["estimated"],
        )
    except Exception:
        logger.warning("rA estimation failed; using true rA", exc_info=True)
        rA_estimated = rA_true

    rA_values = {"true": rA_true, "estimated": rA_estimated}
    logger.info("rA: true=%.4f, estimated=%.4f", rA_true, rA_estimated)

    rC = float(p.rC)
    C1 = float(p.C1)
    C2 = float(p.C2)

    # --- Score all variants: prep once, score per variant ---
    combined = pd.DataFrame({"id": phenotype_df["id"].values})
    combined["generation"] = phenotype_df["generation"].values.astype(np.int32)
    combined["affected1"] = phenotype_df["affected1"].values
    combined["affected2"] = phenotype_df["affected2"].values
    combined["true_A1"] = phenotype_df["A1"].values.astype(np.float32)
    combined["true_A2"] = phenotype_df["A2"].values.astype(np.float32)

    all_metrics: list[dict] = []

    for cip_source in p.cip_sources:
        cip_a1, cip_v1, prev1 = cip_tables[1][cip_source]
        cip_a2, cip_v2, prev2 = cip_tables[2][cip_source]

        # Compute thresholds/w once per CIP source
        thr1, w1 = compute_thresholds_and_w(
            prep.affected1,
            prep.t_observed1,
            cip_a1,
            cip_v1,
            prev1,
        )
        thr2, w2 = compute_thresholds_and_w(
            prep.affected2,
            prep.t_observed2,
            cip_a2,
            cip_v2,
            prev2,
        )
        lookup_thr1, lookup_w1, lookup_thr2, lookup_w2 = build_pheno_lookups(
            prep,
            thr1,
            w1,
            thr2,
            w2,
        )

        for h2_source in p.h2_sources:
            h2_1 = h2_values[1][h2_source]
            h2_2 = h2_values[2][h2_source]

            for rA_source in p.rA_sources:
                rA = rA_values[rA_source]

                for cov_model in p.cov_models:
                    cov_g12 = rA * math.sqrt(h2_1 * h2_2)
                    if cov_model == "genetic":
                        rho_within = cov_g12
                    else:  # genetic_c
                        rho_within = cov_g12 + rC * math.sqrt(C1 * C2)

                    tag = f"{cip_source}_{h2_source}_{rA_source}_{cov_model}"
                    logger.info(
                        "Scoring: %s (h2=%.3f/%.3f, rA=%.3f, rho_w=%.4f)",
                        tag,
                        h2_1,
                        h2_2,
                        rA,
                        rho_within,
                    )

                    scores = score_bivariate_variant(
                        prep,
                        h2_1,
                        h2_2,
                        cov_g12,
                        rho_within,
                        lookup_thr1,
                        lookup_w1,
                        lookup_thr2,
                        lookup_w2,
                    )

                    combined[f"est_1_{tag}"] = scores["est_1"].values.astype(np.float32)
                    combined[f"est_2_{tag}"] = scores["est_2"].values.astype(np.float32)
                    combined[f"var_1_{tag}"] = scores["var_1"].values.astype(np.float32)
                    combined[f"var_2_{tag}"] = scores["var_2"].values.astype(np.float32)
                    combined[f"cov_12_{tag}"] = scores["cov_12"].values.astype(np.float32)
                    combined[f"nrel_{tag}"] = scores["n_relatives"].values.astype(np.int16)

                    metrics = compute_bivariate_metrics(scores)
                    all_metrics.append(
                        {
                            **metrics,
                            "cip_source": cip_source,
                            "h2_source": h2_source,
                            "rA_source": rA_source,
                            "cov_model": cov_model,
                        }
                    )

    scores_path = f"{base_dir}/scores.parquet"
    save_parquet(combined, scores_path)
    logger.info(
        "Wrote bivariate scores: %s (%d rows, %d cols)",
        scores_path,
        len(combined),
        len(combined.columns),
    )

    metrics_path = f"{base_dir}/metrics.tsv"
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    logger.info("Wrote bivariate metrics: %s (%d rows)", metrics_path, len(metrics_df))


try:
    snakemake
    _run_snakemake()
except NameError:
    pass
