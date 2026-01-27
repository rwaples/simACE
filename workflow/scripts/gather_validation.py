"""
Gather validation results from all scenarios into a single TSV file.
"""

import re
import yaml


def extract_metrics(validation_path):
    """Extract key metrics from a validation YAML file."""
    with open(validation_path) as f:
        data = yaml.safe_load(f)

    # Extract scenario and rep from path: results/{scenario}/rep{rep}/validation.yaml
    match = re.search(r"results/([^/]+)/rep(\d+)/validation\.yaml", validation_path)
    if match:
        scenario = match.group(1)
        rep = int(match.group(2))
    else:
        scenario = "unknown"
        rep = 1

    params = data["parameters"]
    summary = data["summary"]

    # Extract key metrics, handling potential missing values
    def get_nested(d, *keys, default=None):
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    row = {
        "scenario": scenario,
        "rep": rep,
        "N": params.get("N"),
        "ngen": params.get("ngen"),
        "A": params.get("A"),
        "C": params.get("C"),
        "E": params.get("E"),
        "p_mztwin": params.get("p_mztwin"),
        "p_nonsocial_father": params.get("p_nonsocial_father"),
        "fam_size": params.get("fam_size"),
        "seed": params.get("seed"),
        "checks_failed": summary.get("checks_failed"),
        "observed_twin_rate": get_nested(data, "twins", "twin_rate", "observed_rate"),
        "variance_A": get_nested(data, "statistical", "variance_A", "observed"),
        "variance_C": get_nested(data, "statistical", "variance_C", "observed"),
        "variance_E": get_nested(data, "statistical", "variance_E", "observed"),
        "mz_twin_A_corr": get_nested(
            data, "heritability", "mz_twin_A_correlation", "observed"
        ),
        "mz_twin_pheno_corr": get_nested(
            data, "heritability", "mz_twin_phenotype_correlation", "observed"
        ),
        "dz_sibling_A_corr": get_nested(
            data, "heritability", "dz_sibling_A_correlation", "observed"
        ),
        "dz_sibling_pheno_corr": get_nested(
            data, "heritability", "dz_sibling_phenotype_correlation", "observed"
        ),
        "half_sib_prop_expected": get_nested(
            data, "half_sibs", "half_sib_pair_proportion", "expected"
        ),
        "half_sib_prop_observed": get_nested(
            data, "half_sibs", "half_sib_pair_proportion", "observed"
        ),
        "offspring_with_half_sib_expected": get_nested(
            data, "half_sibs", "offspring_with_half_sib", "expected"
        ),
        "offspring_with_half_sib_observed": get_nested(
            data, "half_sibs", "offspring_with_half_sib", "observed"
        ),
        "half_sib_A_corr": get_nested(
            data, "half_sibs", "half_sib_A_correlation", "observed"
        ),
        "half_sib_pheno_corr": get_nested(
            data, "half_sibs", "half_sib_phenotype_correlation", "observed"
        ),
        "half_sib_shared_C": get_nested(
            data, "half_sibs", "half_sib_shared_C", "observed"
        ),
        "falconer_h2": get_nested(
            data, "heritability", "falconer_estimate", "observed"
        ),
        "parent_offspring_A_slope": get_nested(
            data, "heritability", "parent_offspring_A_regression", "slope"
        ),
        "parent_offspring_A_r2": get_nested(
            data, "heritability", "parent_offspring_A_regression", "r_squared"
        ),
        "parent_offspring_pheno_slope": get_nested(
            data, "heritability", "parent_offspring_phenotype_regression", "slope"
        ),
        "parent_offspring_pheno_r2": get_nested(
            data, "heritability", "parent_offspring_phenotype_regression", "r_squared"
        ),
    }

    return row


def main(validation_files, output_path):
    """Gather all validation results into a TSV file."""
    rows = []
    for validation_path in validation_files:
        row = extract_metrics(validation_path)
        rows.append(row)

    # Sort by scenario name, then by rep
    rows.sort(key=lambda x: (x["scenario"], x["rep"]))

    # Write TSV
    if rows:
        columns = list(rows[0].keys())
        with open(output_path, "w") as f:
            f.write("\t".join(columns) + "\n")
            for row in rows:
                values = []
                for col in columns:
                    val = row[col]
                    if val is None:
                        values.append("")
                    elif isinstance(val, float):
                        values.append(f"{val:.4g}")
                    else:
                        values.append(str(val))
                f.write("\t".join(values) + "\n")


if __name__ == "__main__":
    validation_files = snakemake.input.validations
    output_path = snakemake.output.tsv

    main(validation_files, output_path)
