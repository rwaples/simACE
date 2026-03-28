data {
  int<lower=1> Nped;                      		// Size of the pedigree
  int<lower=1> Nfam; 			          		// Number of families
  int<lower=1> Nfit;                      		// Number of inds with phenotype data
  int<lower=0> Ndrop;							// difference in Nped - Nfit

  array[Nfit] real y;                     		// standardized phenotype
  array[Nfit] int<lower=1, upper=Nfam> fam;  	// family idx per individual
  array[Nped] int<lower=0, upper=Nped> mother; // idx of mother of each ind
  array[Nped] int<lower=0, upper=Nped> father; // idx of father of each ind
}


parameters {
  real<lower=0> sigma_A;						// sd of additive term
  real<lower=0> sigma_C;						// sd of shared environment term
  real<lower=0> sigma_E;						// sd of personal environment term

  vector[Nped] a_ped;        					// additive genetic effect per individual

  vector[Nfam] c_fam;    						// shared env per family
}


model {
  // Priors on variance components (needed to regularize joint MAP)
  sigma_A ~ normal(0, 1);
  sigma_C ~ normal(0, 1);
  sigma_E ~ normal(0, 1);

  // Genetic effects: founders vs non-founders
  for (i in 1:Nped) {
    if (mother[i] == 0 && father[i] == 0) {
      // founder
      a_ped[i] ~ normal(0, sigma_A);
    } else {
      real mp = 0;
      int n_par = 0;
      if (mother[i] > 0) {
        mp += a_ped[mother[i]];
        n_par += 1;
      }
      if (father[i] > 0) {
        mp += a_ped[father[i]];
        n_par += 1;
      }
      mp /= n_par;                  // midparent
      // std of Mendelian sampling term: sigma_A / sqrt(2)
      a_ped[i] ~ normal(mp, sigma_A / sqrt(2));
    }
  }


  // Shared environment: family random effects
  c_fam ~ normal(0, sigma_C);

  // select the subset of data used for fitting the model
  vector[Nfit] c_fit;
  vector[Nfit] a_fit;

  for (i in 1:Nfit) {
    c_fit[i] = c_fam[fam[i]];
  }

  for (i in 1:Nfit) {
    a_fit[i] = a_ped[i+Ndrop];
  }

  // Profile out mu (REML: mean is not a parameter)
  real mu_hat = mean(to_vector(y) - a_fit - c_fit);

  // Likelihood
  y ~ normal(mu_hat + a_fit + c_fit, sigma_E);

  // REML correction: -0.5 * log(X' V^{-1} X) = -0.5*log(Nfit) + log(sigma_E)
  target += -0.5 * log(Nfit) + log(sigma_E);
}


generated quantities {
  real V_A = square(sigma_A);
  real V_C = square(sigma_C);
  real V_E = square(sigma_E);
  real V_T = V_A + V_C + V_E;
  real h2 = V_A / V_T;
  real c2 = V_C / V_T;
  real e2 = V_E / V_T;
}
