data {
  int<lower=1> Nped;                      		// Size of the pedigree
  int<lower=1> Nfam; 			          		// Number of families
  int<lower=1> Nfit;                      		// Number of inds with phenotype data
  int<lower=0> Ndrop;							// difference in Nped - Nfit
  
  array[Nfit] real y;                     		// standardized phenotype
  array[Nfit] int<lower=1, upper=Nfam> fam;  	// family idx per individual
  array[Nped] int<lower=0, upper=Nped> mother; // idx of mother of each ind
  array[Nped] int<lower=0, upper=Nped> father; // idx of father of each ind
  array[Nped] real<lower=0> dii;               // Henderson's D-inverse diagonal
}


parameters {
  real mu;                                      // mean phenotype
  
  real<lower=0> sigma_A;						// sd of additive term
  real<lower=0> sigma_C;						// sd of shared environment term
  real<lower=0> sigma_E;						// sd of personal environment term
  
  vector[Nped] a_ped;        					// additive genetic effect per individual

  vector[Nfam] c_fam;    						// shared env per family
}


model {
  // Priors
  mu      ~ normal(0, 0.01);
  sigma_A ~ normal(0, 1);
  sigma_C ~ normal(0, 1);
  sigma_E ~ normal(0, 1);
  
 
  // Genetic effects: Henderson's decomposition A^{-1} = (I-P)' D^{-1} (I-P)
  for (i in 1:Nped) {
    real mp = 0;
    if (mother[i] > 0) mp += a_ped[mother[i]];
    if (father[i] > 0) mp += a_ped[father[i]];
    mp *= 0.5;
    a_ped[i] ~ normal(mp, sigma_A / sqrt(dii[i]));
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

  // Likelihood
  y ~ normal(a_fit + c_fit, sigma_E);

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

