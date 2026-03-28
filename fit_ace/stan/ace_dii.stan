data {
  int<lower=1> N;                     // number of individuals in ped
  array[N] int<lower=0> mother;       // 0 if unknown, else 1..N - indexes of mother
  array[N] int<lower=0> father;       // 0 if unknown, else 1..N - indexes of father
  vector<lower=0>[N] dii;             // dii

  int<lower=1> Npheno;                // number of phenotyped individuals
  array[Npheno] int<lower=1, upper=N> pid;  // indices of phenotyped individuals
  vector[Npheno] y;

  int<lower=1> H;                     // number of households
  array[Npheno] int<lower=1, upper=H> household;

  int<lower=1> G;                        // number of generations (tiers)
  array[G + 1] int<lower=1> tier_start;  // tier k spans [tier_start[k], tier_start[k+1]-1]
}


transformed data{
  vector<lower=0>[N] sqrt_dii = sqrt(dii);
}


parameters {
  real mu;
  real<lower=0> sigma_A;
  real<lower=0> sigma_E;
  real<lower=0> sigma_H;
  vector[H] z_h;                       // noncentered household
  vector[N] z_m;                       // noncentered mendelian sampling
}


transformed parameters {
  vector[H] u;
  vector[N] m;
  vector[N] a;

  profile("household") {
    u = sigma_H * z_h;
  }

  profile("mendelian") {
    // Mendelian sampling terms m_i ~ N(0, sigma_A^2 * dii_i)
    m = sigma_A * (sqrt_dii .* z_m);
  }

  profile("recursion") {
    // Founders: breeding value = mendelian sampling only
    a[tier_start[1] : tier_start[2] - 1] = m[tier_start[1] : tier_start[2] - 1];

    // Non-founder tiers: vectorized midparent + mendelian
    for (k in 2:G) {
      int s = tier_start[k];
      int e = tier_start[k + 1] - 1;
      a[s:e] = 0.5 * (a[mother[s:e]] + a[father[s:e]]) + m[s:e];
    }
  }
}


model {
  profile("priors") {
    mu ~ normal(0, 2);
    sigma_A ~ normal(0, 1);
    sigma_E ~ normal(0, 1);
    sigma_H ~ normal(0, 1);

    z_h ~ std_normal();
    z_m ~ std_normal();
  }

  profile("likelihood") {
    y ~ normal(mu + a[pid] + u[household], sigma_E);
  }
}


generated quantities {
  real V_A = square(sigma_A);
  real V_H = square(sigma_H);
  real V_E = square(sigma_E);
  real V_T = V_A + V_H + V_E;
  real h2 = V_A / V_T;
  real c2 = V_H / V_T;
  real e2 = V_E / V_T;
}
