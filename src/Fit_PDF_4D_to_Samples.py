
# coding: utf-8

# In[1]:

# sample_start = 0
# sample_length = 1000
# data_directory = '../data/'
# class_system = 'IPC4'
# rescale_z = True
# relatedness_types = []


# In[2]:

import pandas as pd
from pylab import *


# In[4]:

popularity_types = ['Class_Patent_Count_1_years_Previous_Year_Percentile']


# In[5]:

from time import time
t = time()


# In[7]:

entry_data = pd.HDFStore((data_directory+
                             'Agent_Entries/samples/agent_entry_data_%s_sample_%i_%i_agents.h5'%(class_system,
                                                                                                     sample_start,
                                                                                                     sample_start+sample_length)))
all_data = entry_data['all_available_classes']
entry_data.close()

for relatedness in relatedness_types:
    print(relatedness)
    if 'z_score' in relatedness:
        if rescale_z:
            all_data.ix[all_data[relatedness]<0, relatedness] = 0
            f = lambda x: ((x-1)/(x+1))/2+.5
            all_data[relatedness] = f(all_data[relatedness].values)#*100
            n_bins = 25.0
            all_data.ix[all_data[relatedness]==0, relatedness] = -1
        else:
            n_bins=500.0
    elif 'percent_positive' in relatedness:
        n_bins=25.0
#         all_data[relatedness] *= 100
        #Flag the values that are 0 as -1 so they are below the range of the bins when digitizing.
        #They will thus be labeled as "0" when digitizing, so we know they were literally 0, and not just close to 0.
        all_data.ix[all_data[relatedness]==0, relatedness] = -1

    all_data[relatedness] = digitize(all_data[relatedness], arange(0,1, 1/n_bins))/n_bins

for popularity in popularity_types:
    n_bins=500.0    
    all_data[popularity] /=100
    all_data[popularity] = digitize(all_data[popularity], arange(0,1, 1/n_bins))/n_bins

all_data['Agent_Previous_Citations_to_Class'] = (all_data['Agent_Previous_Citations_to_Class']>0).astype('uint8')
all_data['CoAgent_Count_in_Class'] = (all_data['CoAgent_Count_in_Class']>0).astype('uint8')
all_data = all_data[all_data['Application_Year']>1976]
all_data['Application_Year'] = all_data['Application_Year'].astype('int')


# In[8]:

print("%.1f minutes to load data."%((time()-t)/60))


# In[9]:

import os
def create_directory_if_not_existing(f):
    try:
        os.makedirs(f)
    except OSError:
        pass

f = data_directory+'Predictive_Models/PDF/samples/agent_entry_data_%s_sample_%i_%i_agents/'%(class_system,
                                                                                                     sample_start,
                                                                                                     sample_start+sample_length)
create_directory_if_not_existing(f)


# In[10]:

for relatedness in relatedness_types:
    print(relatedness)
    
    g = f+relatedness
    if 'z_score' in relatedness and rescale_z: 
        g += '_rescaled'
    g += '/'
    create_directory_if_not_existing(g)

    for popularity in popularity_types:
        h = g+popularity+'/'
        create_directory_if_not_existing(h)
        data = all_data.groupby(['Application_Year', 
                                 relatedness, 
                                 popularity,
                                 'Agent_Previous_Citations_to_Class',
                                 'CoAgent_Count_in_Class'])['Patent'].count()
        entries = all_data[all_data['Entered']>0].groupby(['Application_Year', 
                                                           relatedness, 
                                                           popularity,
                                                          'Agent_Previous_Citations_to_Class',
                                                          'CoAgent_Count_in_Class'])['Patent'].count()
        data.to_hdf(h+'data.h5', 'data', complib='blosc', complevel=9)
        entries.to_hdf(h+'data.h5', 'entries', complib='blosc', complevel=9)


# In[11]:

print("%.1f minutes to load and write data."%((time()-t)/60))

