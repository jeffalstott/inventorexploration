
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *
import gc
from time import time


# In[2]:

# data_directory = '../data/'
# class_system = 'IPC4'
# sample_start = 0
# max_sample = 773000
# sample_length = 1000
# n_combine = 100000
# rescale_z = True
# relatedness_types = []


# In[4]:

pdfs = {}

for relatedness in relatedness_types:
    pdfs[relatedness] = {}
    for popularity in popularity_types:
        pdfs[relatedness][popularity] = {}
        for m in ['data', 'entries']:
            pdfs[relatedness][popularity]['data'] = []
            pdfs[relatedness][popularity]['entries'] = []


# In[5]:

def add_sparse_series(list_of_series):
    return pd.concat(list_of_series).groupby(level=arange(shape(list_of_series[0].index.levels)[0]).tolist()).sum()


# In[ ]:

t0 = time()

t = time()
while sample_start<max_sample:
    
    f = data_directory+ 'Predictive_Models/PDF/samples/agent_entry_data_%s_sample_%i_%i_agents/'%(class_system,
                                                                                                         sample_start,
                                                                                                         sample_start+sample_length)              
    for relatedness in relatedness_types:
        g = f+relatedness
        if 'z_score' in relatedness and rescale_z: 
            g += '_rescaled'
        g += '/'
        for popularity in popularity_types:
            h = g+popularity+'/'
            pdfs[relatedness][popularity]['data'].append(pd.read_hdf(h+'data.h5', 'data'))
            pdfs[relatedness][popularity]['entries'].append(pd.read_hdf(h+'data.h5', 'entries'))

    gc.collect()
    if not sample_start%(n_combine) and sample_start!=0:
        print("%.1f minutes to access %i samples (now at %i)"%(((time()-t)/60), n_combine, sample_start))
        t = time()
        for relatedness in relatedness_types:
            for popularity in popularity_types:
                pdfs[relatedness][popularity]['data'] = [add_sparse_series(pdfs[relatedness][popularity]['data'])]
                pdfs[relatedness][popularity]['entries'] = [add_sparse_series(pdfs[relatedness][popularity]['entries'])]    
        gc.collect()
        print("%.1f minutes to combine %i samples (now at %i)"%(((time()-t)/60), n_combine, sample_start))
        t = time()
    sample_start += sample_length
print("%.1f minutes to access samples"%((time()-t0)/60))


# In[ ]:

t = time()
for relatedness in relatedness_types:
    for popularity in popularity_types:
        pdfs[relatedness][popularity]['data'] = add_sparse_series(pdfs[relatedness][popularity]['data'])
        pdfs[relatedness][popularity]['entries'] = add_sparse_series(pdfs[relatedness][popularity]['entries'])
print('%.1f to combine last samples'%(time()-t))


# In[ ]:

def cumulate(s):
    return s.sort_index(level='Application_Year').groupby(level=[1,2,3,4]).cumsum()

import os
def create_directory_if_not_existing(f):
    try:
        os.makedirs(f)
    except OSError:
        pass


# In[18]:

create_directory_if_not_existing(data_directory+'Predictive_Models/')

f = data_directory+'Predictive_Models/PDF/'
for relatedness in relatedness_types:
    g = f+relatedness
    if 'z_score' in relatedness and rescale_z: 
        g += '_rescaled'
    g += '/'
    create_directory_if_not_existing(f)
    for popularity in popularity_types:
        h = g+popularity+'/'
        create_directory_if_not_existing(h)
        pdfs[relatedness][popularity]['data'].to_hdf(h+'year_by_year.h5', 'data', complib='blosc', complevel=9)
        pdfs[relatedness][popularity]['entries'].to_hdf(h+'year_by_year.h5', 'entries', complib='blosc', complevel=9)
        
        store = pd.HDFStore(h+'cumulative.h5', complib='blosc', complevel=9)
        for year in arange(1977,2010+1):
            e_this_year = pdfs[relatedness][popularity]['entries'].ix[:year].groupby(level=[1,2,3,4]).sum()
            d_this_year = pdfs[relatedness][popularity]['data'].ix[:year].groupby(level=[1,2,3,4]).sum()
            store.put('entries/year_%i'%year, e_this_year)
            store.put('data/year_%i'%year, d_this_year)
            store.put('pdf/year_%i'%year, (e_this_year/d_this_year).fillna(0))
        store.close()

