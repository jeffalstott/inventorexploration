
# coding: utf-8

# In[1]:

# data_directory = '../data/'
# from os import path
# abs_path_data_directory = path.abspath(data_directory)+'/'

# cmdstan_directory = path.abspath('cmdstan-2.9.0/')+'/'


# In[14]:

from os import system, makedirs
def create_directory_if_not_existing(f):
    try:
        makedirs(f)
    except OSError:
        pass    
    
model_directory = abs_path_data_directory+'Performance_Models/'
create_directory_if_not_existing(model_directory)


# In[3]:

with open(model_directory+'single_counts_sampling_model.stan', 'w') as f:
    f.write("""
data {
  int<lower=1> N;                // number of data points
  int<lower=0> y[N];   // outcomes
  int<lower=1> K;                // number parameters 
  matrix[N,K] x;                 // individual predictors
  vector[N] baseline; 
}

parameters {
  vector[K] beta;
  real<lower=0> mu_phi;
  real<lower=0> var_phi;
  real<lower=0> phi;
}

model {
  beta ~ normal(0,2);
  mu_phi ~ normal(0,2);//cauchy(0,2.5);
  var_phi ~ normal(0,2); //cauchy(0,2.5);
  phi ~ gamma(mu_phi^2/var_phi,mu_phi/var_phi);
  y ~ neg_binomial_2_log(x*beta + log(baseline), phi);  
}"""
           )


# In[4]:

with open(model_directory+'hits_sampling_model.stan', 'w') as f:
    f.write("""
data {
  int<lower=1> N;                // number of data points
  int<lower=0, upper=1> y[N];   // outcomes
  int<lower=1> K;                // number parameters 
  matrix[N,K] x;                 // individual predictors
}

parameters {
  vector[K] beta;
}

model {
  beta ~ normal(0,5);
  y ~ bernoulli_logit(x*beta);  
}"""
           )


# In[15]:

from os import system

system("make -C %s %s"%(cmdstan_directory,
                       model_directory+'single_counts_sampling_model'))

system("make -C %s %s"%(cmdstan_directory,
                       model_directory+'hits_sampling_model'))

