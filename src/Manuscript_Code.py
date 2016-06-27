
# coding: utf-8

# In[1]:

import pandas as pd
# %pylab inline
from pylab import *


# In[2]:

data_directory = '../data/'
cmdstan_directory = 'cmdstan-2.9.0/'


# In[3]:

from os import path
import sys
abs_path_data_directory = path.abspath(data_directory)+'/'
python_location = path.dirname(sys.executable)+'/python'
cmdstan_directory = path.abspath(cmdstan_directory)+'/'
### Necessary for submitting analyses to a cluster


# In[3]:

import os
def create_directory_if_not_existing(f):
    try:
        os.makedirs(f)
    except OSError:
        pass


# Organize data for citations, co-classifications and occurrences
# ===

# In[9]:

# print("Organizing Citations")
# %run -i Organize_Citations.py
# print("Organizing Classifications")
# %run -i Organize_Classifications.py
# print("Organizing Occurrences")
# %run -i Organize_Occurrences.py


# Define parameters
# ===

# Define classes and entities to analyze
# ---

# In[4]:

class_systems = ['IPC4']
occurrence_entities = {#'Firm': ('occurrences_organized.h5', 'entity_classes_Firm'),
                       'Inventor': ('occurrences_organized.h5', 'entity_classes_Inventor'),
                       'PID': ('classifications_organized.h5', 'patent_classes'),
                       }
entity_types = list(occurrence_entities.keys())


# Define what years to calculate networks for
# ---

# In[5]:

target_years = 'all'


# Define number of years of history networks should include
# ---

# In[9]:

all_n_years = ['all', 1]

def create_n_years_label(n_years):
    if n_years is None or n_years=='all' or n_years=='cumulative':
        n_years_label = ''
    else:
        n_years_label = '%i_years_'%n_years
    return n_years_label


# Calculate empirical networks
# ===

# In[ ]:

citation_metrics = ['Class_Cites_Class_Count',
                    'Class_Cited_by_Class_Count']


# In[12]:

create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/')


# In[15]:

# ### Create empirical networks
# randomized_control = False

# for class_system in class_systems:
#     for n_years in all_n_years:
#         print("Calculating for %s------"%class_system)
#         print("Calculating for %s years------"%str(n_years))
#         ### Calculate citation networks
#         %run -i Calculating_Citation_Networks.py
#         all_networks = networks

#         ### Calculate co-occurrence networks
#         preverse_years = True
#         for entity_column in entity_types:
#             target_years = 'all'
#             print(entity_column)
#             occurrence_data, entity_data = occurrence_entities[entity_column]
#             %run -i Calculating_CoOccurrence_Networks.py
#             all_networks.ix['Class_CoOccurrence_Count_%s'%entity_column] = networks

#         ind = ['Class_CoOccurrence_Count_%s'%entity for entity in entity_types]
#         store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/class_relatedness_networks_cooccurrence.h5', 
#                         mode='a', table=True)
#         n_years_label = create_n_years_label(n_years)
#         store.put('/empirical_cooccurrence_%s%s'%(n_years_label,class_system), all_networks.ix[ind], 'table', append=False)
#         store.close()

#         #### Combine them both
#         store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5', 
#                             mode='a', table=True)
#         store.put('/empirical_'+n_years_label+class_system, all_networks, 'table', append=False)
#         store.close()


# Calculate randomized, synthetic networks
# ====

# Make directories
# ---

# In[11]:

create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/controls/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/')


# Run randomizations
# ---
# (Currently set up to use a cluster)

# In[12]:

first_rand_id = 0
n_randomizations = 1000
overwrite = False


# In[13]:

# create_directory_if_not_existing('jobfiles/')

# for class_system in class_systems:
#     for n_years in all_n_years:
#         ### Citations
#         create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
#         basic_program = open('Calculating_Citation_Networks.py', 'r').read()
#         job_type = 'citations'
#         options="""class_system = %r
# target_years = %r
# n_years = %r
# data_directory = %r
# randomized_control = True
# citation_metrics = %r
#     """%(class_system, target_years, n_years, abs_path_data_directory, citation_metrics)

#         %run -i Calculating_Synthetic_Networks_Control_Commands

#         ### Co-occurrences
#         create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)
#         basic_program = open('Calculating_CoOccurrence_Networks.py', 'r').read()
#         job_type = 'cooccurrence'
#         for entity in entity_types:
#             occurrence_data, entity_data = occurrence_entities[entity]
#             options = """class_system = %r
# target_years = %r
# n_years = %r
# data_directory = %r
# randomized_control = True
# preserve_years = True
# chain = False
# occurrence_data = %r
# entity_data = %r
# entity_column = %r
# print(occurrence_data)
# print(entity_data)
# print(entity_column)
#     """%(class_system, target_years, n_years, abs_path_data_directory, occurrence_data, entity_data, entity)

#             %run -i Calculating_Synthetic_Networks_Control_Commands


# Integrate randomized data and calculate Z-scores
# ---
# Note: Any classes that have no data (i.e. no patents within that class) will create z-scores of 'nan', which will be dropped when saved to the HDF5 file. Therefore, the z-scores data will simply not includes these classes.

# In[17]:

# n_controls = n_randomizations

# output_citations = 'class_relatedness_networks_citations'
# # output_citations = False
# output_cooccurrence = 'class_relatedness_networks_cooccurrence'
# # output_cooccurrence = False
# combine_outputs = True


# for class_system in class_systems:
#     print(class_system)
#     for n_years in all_n_years:
#         print(n_years)
#         n_years_label = create_n_years_label(n_years)
#         cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_preserve_years_%s'

#         %run -i Calculating_Synthetic_Networks_Integrate_Runs.py


# Delete individual runs of randomized data
# ---

# In[11]:

# from shutil import rmtree

# for class_system in class_systems:
#     rmtree(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
#     rmtree(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)  


# Regress out popularity from relatedness measures
# ---
# First create popularity-by-year networks for all class systems and n_years

# In[8]:

# %run -i Calculating_Popularity_Networks.py


# In[13]:

# %run -i Regressing_Popularity_Out_of_Z_Scores.py


# Create inventor entries data
# ----

# Precalculate some data that all runs will rely on and which takes a long time to calculate

# In[ ]:

sample_length = 1000


# In[39]:

# for class_system in class_systems:
#     %run -i Organize_Inventor_Patent_Data.py


# In[38]:

# for class_system in class_systems:
#     %run -i Calculating_Supporting_Data_for_Agent_Entry_Calculations.py


# In[ ]:

# overwrite = False

# create_directory_if_not_existing('jobfiles/')
# create_directory_if_not_existing(data_directory+'Agent_Entries/')
# create_directory_if_not_existing(data_directory+'Agent_Entries/samples/')

# for class_system in class_systems:

#     store = pd.HDFStore(data_directory+'organized_patent_data.h5')
#     agents_lookup_explorers = store['agents_lookup_explorers_%s'%class_system]
#     store.close()

#     n_agents = agents_lookup_explorers.shape[0]


#     all_samples_end = n_agents+sample_length
#     all_samples_start = 0
#     %run -i Calculating_Inventor_Entries_Data_Control_Commands.py


# Predict inventor entries
# ===

# In[ ]:

relatedness_types = [ 'Class_Cites_Class_Count_1_years_percent_positive_all_years_back_mean',
                     'Class_Cited_by_Class_Count_1_years_percent_positive_all_years_back_mean',
                     'Class_CoOccurrence_Count_Inventor_1_years_percent_positive_all_years_back_mean',
                     'Class_CoOccurrence_Count_PID_1_years_percent_positive_all_years_back_mean',
                    ]

rescale_z = True #Doesn't actually matter if you're not using z-score based R, but it is necessary to assign


# Calculate PDFs for each sample of inventors

# In[ ]:

# create_directory_if_not_existing('jobfiles/')

# for class_system in class_systems:

#     store = pd.HDFStore(data_directory+'organized_patent_data.h5')
#     agents_lookup_explorers = store['agents_lookup_explorers_%s'%class_system]
#     store.close()

#     n_agents = agents_lookup_explorers.shape[0]

#     all_samples_end = n_agents#+sample_length
#     all_samples_start = 0
#     samples = range(all_samples_start,all_samples_end,sample_length)
    
#     rescale_z = True
#     %run -i Fit_PDF_4D_to_Samples_Control_Commands.py


# Integrate the PDFs

# In[ ]:

# create_directory_if_not_existing('jobfiles/')
# for class_system in class_systems:

#     store = pd.HDFStore(data_directory+'organized_patent_data.h5')
#     agents_lookup_explorers = store['agents_lookup_explorers_%s'%class_system]
#     store.close()

#     n_agents = agents_lookup_explorers.shape[0]

#     max_sample = n_agents
#     sample_start = 0
#     n_combine = 100000

#     %run -i Fit_PDF_4D_to_Samples_Integrate_Control_Commands.py


# Use the PDFs to predict the samples

# In[ ]:

# create_directory_if_not_existing('jobfiles/')
# for class_system in class_systems:

#     store = pd.HDFStore(data_directory+'organized_patent_data.h5')
#     agents_lookup_explorers = store['agents_lookup_explorers_%s'%class_system]
#     store.close()

#     n_agents = agents_lookup_explorers.shape[0]
#     all_samples_end = n_agents#+sample_length
#     all_samples_start = 0
#     samples = range(all_samples_start,all_samples_end,sample_length)

#     %run -i Predict_Samples_NB_4D_Control_Commands.py


# Integrate the data on entries with predictions

# In[ ]:

# target = 'entries_with_predictions_NB_4D'

# for class_system in class_systems:
#     %run -i Calculating_Inventor_Entries_Data_Integrate_Runs.py


# Performance modeling
# ====

# In[ ]:

target = 'entries'


# Calculate inventors' performance after entry

# In[ ]:

# for class_system in class_systems:
#     %run -i Calculating_Performance_Data.py


# Create and compile performance models with Stan

# In[11]:

# %run -i Stan_Models.py


# Use HMC sampling to do Bayesian inference

# In[ ]:

# n_chains = 50
# chains_start = 0
# num_warmup = 300
# num_samples = 300
# n_observations = 'full'#10000

# relatedness_type = 'Class_Cited_by_Class_Count'
# relatedness = '%s_1_years_percent_positive_all_years_back_mean'%relatedness_type
# popularity = 'Class_Patent_Count_1_years_Previous_Year_Percentile'

# count_variables = ['Agent_Number_of_Patents_in_Class','Agent_Number_of_Citations_in_Class']
# hit_thresholds = [3,4,5,6]

# for class_system in class_systems:
#     %run -i Stan_Sampling.py


# Integrate the samples together 

# In[ ]:

get_ipython().magic('run -i Stan_Sampling_Integrate.py')


# Make figures
# ===

# In[ ]:

# figures_directory = '../manuscript/figs/'
# save_as_manuscript_figures = True

# for class_system in class_systems:
#     %run -i Manuscript_Figures.py

