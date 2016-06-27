
# coding: utf-8

# In[1]:

# n_chains = 50
# chains_start = 0
# num_warmup = 300
# num_samples = 300
# n_observations = 'full'#100000
# data_directory = '../data/'
# from os import path
# abs_path_data_directory = path.abspath(data_directory)+'/'
# target = 'entries'
# class_system = 'IPC4'

# relatedness_type = 'Class_Cited_by_Class_Count'
# relatedness = '%s_1_years_percent_positive_all_years_back_mean'%relatedness_type
# popularity = 'Class_Patent_Count_1_years_Previous_Year_Percentile'


# hit_thresholds = [3,4,5,6]

# count_variables = ['Agent_Number_of_Patents_in_Class','Agent_Number_of_Citations_in_Class']


# In[2]:

run_label = n_observations

import os
def create_directory_if_not_existing(f):
    try:
        os.makedirs(f)
    except OSError:
        pass
model_directory = abs_path_data_directory+'Performance_Models/'
create_directory_if_not_existing(model_directory+'stan_samples/')
create_directory_if_not_existing(model_directory+'stan_samples/{0}/'.format(run_label))


# In[4]:

models_store = pd.HDFStore(model_directory+'performance_models.h5')


# In[55]:

from os import system

def submit_cmdstan_jobs(model,data_file):
    for chain in range(chains_start,chains_start+n_chains):
        print(chain)
        header = """#!/usr/bin/env bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=72:00:00
#PBS -l mem=4000m
#PBS -N chain_{0}_{1}
""".format(data_file, chain)

        this_program = ("{0}{1} "
               "sample num_samples={3} num_warmup={4} "
               "data file={0}{2} init=.5 "
               "output file={0}stan_samples/{6}/output_{2}_{5}.csv".format(model_directory, 
                                                             model,
                                                             data_file,
                                                             num_samples, num_warmup,
                                                                           chain, run_label))
        this_program = header+this_program
        this_job_file = 'jobfiles/chain_{0}_{1}'.format(data_file, chain)


        f = open(this_job_file, 'w')
        f.write(this_program)
        f.close()

        system('qsub '+this_job_file)


# In[11]:

store = pd.HDFStore(data_directory+'Agent_Entries/agent_%s_%s.h5'%(target, class_system), mode='r')
entries = store['%s_%s_with_performance'%(target, class_system)]
store.close()

entries['Relatedness'] = entries[relatedness]
entries['Popularity']  = entries[popularity]

if entries['Relatedness'].max()==100:
    entries['Relatedness'] /= 100 

if entries['Popularity'].max()==100:
    entries['Popularity'] /= 100 

entries['Years_to_2010'] = 2010-entries['Application_Year']

entries = entries.ix[entries['Years_Since_First_Patent']<40]
years_since_first = entries['Years_Since_First_Patent'].values
years_since_first[years_since_first==0] = 1
entries['Agent_Productivity_Patents'] = entries['Agent_Patent_Number']/years_since_first

entries = entries[entries['Application_Year']>1976]
entries = entries[entries['Application_Year']<=2005]


for c in count_variables:
    b = c+'_Mean_for_Year_and_Class_of_New_Immigrants_to_Class'
    if entries[c].min()==1:
        entries[c] -= 1
        entries[b] -= 1
    entries = entries[entries[b]>0]


def zscore_in_group(x, performance, reference_group):
    return ((x[performance]-x[performance+'_Mean_'+reference_group])/
                             x[performance+'_STD_'+reference_group])
def high_zscore_in_group(x, performance, reference_group, thr):
    return (zscore_in_group(x, performance, reference_group)>thr)

for thr in hit_thresholds:
    entries['Citations_Hit_%i'%thr] = high_zscore_in_group(entries, 
                                                    'First_Patent_Citations', 
                                                    'for_Year_and_Class', thr).astype('int')
    
    
entries.to_hdf(model_directory+'entries_for_performance_analysis.h5', 'entries')


# In[56]:

if n_observations=='full' or n_observations=='all':
    N = entries.shape[0]
else:
    N = n_observations
from numpy.random import choice
ind = choice(entries.index,N, False)


# In[57]:

from patsy import dmatrix
from pystan import stan_rdump


# In[59]:

model = 'single_counts_sampling_model'
hdf_label = 'counts'
formula_variables = ['Relatedness',
                     'np.power(Relatedness, 2)',
                     'Popularity',
                     'np.power(Popularity, 2)',
                     'log(Agent_Previous_Citations_to_Class+1)',
                     'log(Agent_Productivity_Patents)',
                     'log(CoAgent_Previous_Patent_Count_in_Class+1)']
formula = " + ".join(formula_variables)
models_store['%s/formula_variables'%hdf_label] = pd.Series(formula_variables)


for count_variable in count_variables:
    baseline = count_variable+'_Mean_for_Year_and_Class_of_New_Immigrants_to_Class'

    predictors = dmatrix(formula, entries.ix[ind])
    stan_data = {'y': asarray(entries.ix[ind, count_variable].astype('int')),
                 'x': asarray(predictors),
                 'N': N,
                 'K': predictors.shape[1],
                 'baseline': asarray(entries.ix[ind,baseline])
             }
            
    data_file = 'counts_data_{0}_{1}.stan'.format(count_variable, n_observations)
    stan_rdump(stan_data, model_directory+data_file)
    submit_cmdstan_jobs(model,data_file)


# In[ ]:

model = 'hits_sampling_model'
hdf_label = 'hits'

formula_variables = ['Relatedness',
                     'np.power(Relatedness, 2)',
                     'np.power(Relatedness, 3)',                     
                     'Popularity',
                     'np.power(Popularity, 2)',
                     'log(Agent_Previous_Citations_to_Class+1)',
                     'log(Agent_Productivity_Patents)',
                     'log(CoAgent_Previous_Patent_Count_in_Class+1)']
formula = " + ".join(formula_variables)

models_store['%s/formula_variables'%hdf_label] = pd.Series(formula_variables)

for threshold in hit_thresholds:
    predictors = dmatrix(formula, entries.ix[ind])
    stan_data = {'y': asarray(entries.ix[ind, 'Citations_Hit_%i'%threshold]),
                 'x': asarray(predictors),
                 'N': N,
                 'K': predictors.shape[1],
             }
            
    data_file = 'hits_data_thr_{0}_{1}.stan'.format(threshold, n_observations)
    stan_rdump(stan_data, model_directory+data_file)
    submit_cmdstan_jobs(model,data_file)


# In[ ]:

models_store.close()

