import glob

import numpy as np
import pandas as pd

# import graph_tool.all as gt
# import seaborn as sns
from cmdstanpy import CmdStanModel

# ## Notes to consider
# - Fit a model with and without the shared environmental component
# - Fit with different seeds to look at CI coverage, bias and consistency
# - What about a qualitative trait vs quatitative trait?
# - Variables worth considering:
#   - A, C, E components
#   - Number of generations with pedigree data
#   - Number of generations with phenotype information
#   - POPSIZE
#   - Genomic architecture
#   - Distribution of family sizes
#   - Rates of half-sibs (HS) and MZ twins


peds = glob.glob("/home/ryanw/parent_of_origin/ACE/sim/ace.gen_*.990.individuals.tsv")
peds


def read_ped(path):
    df = pd.read_csv(path, sep="\t")
    df["id"] = df["id"].astype(int)
    df["mother"] = df["mother"].astype(int)
    df["father"] = df["father"].astype(int)
    df["familyID"] = df["familyID"].astype(int)
    df["twin"] = df["twin"].astype(int)
    return df


ped = pd.concat([read_ped(x) for x in peds])
ped = ped.sort_values("id", ascending=True).reset_index(drop=True)


# ## Set up the pedigree fit model


Nped = len(ped)
Nfit = 5000
_, fam_idx = np.unique(ped.iloc[-Nfit:].familyID, return_inverse=True)
fam = fam_idx + 1  # Stan wants 1-based indexes
Nfam = len(np.unique(fam))


# standardize the phenotpye
phenotype = ped.phenotype
y = phenotype[-Nfit:].values
ystd = np.std(y)
if ystd > 0:
    y = (y - np.mean(y)) / ystd
else:
    y = y - np.mean(y)
y.mean(), y.std()


## need to be one-based indexes for stan
## hence adding one
idx_of_id = pd.Series(ped.index, index=ped.id)
mother = (ped.mother.map(idx_of_id).fillna(-1).astype(int) + 1).values
father = (ped.father.map(idx_of_id).fillna(-1).astype(int) + 1).values


# data structure for Stan

pedigree_data = {
    "Nped": Nped,
    "Nfam": Nfam,
    "Nfit": Nfit,
    "Ndrop": Nped - Nfit,
    "y": y,
    "fam": fam,
    "mother": mother,
    "father": father,
}


# compile the Stan model
ace_pedigree_model = CmdStanModel(
    stan_file="/home/ryanw/parent_of_origin/ACE/stan/fit_pedigree_ace.stan", cpp_options={"STAN_THREADS": "true"}
)


# fit the pedigree model
fit_pedigree = ace_pedigree_model.sample(
    data=pedigree_data,
    chains=4,
    parallel_chains=4,
    threads_per_chain=4,
    seed=129,
    thin=2,
    iter_sampling=10000,
    max_treedepth=14,
    inits={
        "mu": 0,
        "sigma_A": np.sqrt(0.4),
        "sigma_C": np.sqrt(0.3),
        "sigma_E": np.sqrt(0.3),
    },
    show_console=False,
)


sum_ped.to_csv("/home/ryanw/parent_of_origin/ACE/stan/fit/ace.990.fit_summary.csv", sep="\t", index=True)
