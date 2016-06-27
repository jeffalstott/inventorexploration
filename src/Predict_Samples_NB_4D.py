
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *
import gc


# In[2]:

# data_directory = '../data/'
# class_system = 'IPC4'
# sample_start = 0
# sample_length = 1000
# relatedness_types = []


# In[3]:

model_years = arange(1980, 2010, 10)


# In[6]:

popularity_types = ['Class_Patent_Count_1_years_Previous_Year_Percentile']


# In[7]:

from time import time


# In[9]:

t = time()
data = {}
entries = {}
pdfs = {}


dummy_relatedness = relatedness_types[0]
dummy_popularity = popularity_types[0]

f = data_directory+'Predictive_Models/PDF/%s/%s/'%(dummy_relatedness,dummy_popularity)
for variable in ['Agent_Previous_Citations_to_Class', 'CoAgent_Count_in_Class']:
    data[variable] = {}
    entries[variable] = {}
    pdfs[variable] = {}
    for year in model_years:
        data[variable][year] = pd.read_hdf(f+'cumulative.h5', 'data/year_%i'%year).astype('float32')
        entries[variable][year] = pd.read_hdf(f+'cumulative.h5', 'entries/year_%i'%year).astype('float32')
        data[variable][year] = data[variable][year].groupby(level=variable).sum()
        entries[variable][year] = entries[variable][year].groupby(level=variable).sum()
        pdfs[variable][year] = (entries[variable][year]/data[variable][year]).fillna(0)


for variable in relatedness_types:
    f = data_directory+'Predictive_Models/PDF/%s/%s/'%(variable,dummy_popularity)
    data[variable] = {}
    entries[variable] = {}
    pdfs[variable] = {}
    for year in model_years:
        data[variable][year] = pd.read_hdf(f+'cumulative.h5', 'data/year_%i'%year).astype('float32')
        entries[variable][year] = pd.read_hdf(f+'cumulative.h5', 'entries/year_%i'%year).astype('float32')
        data[variable][year] = data[variable][year].groupby(level=variable).sum()
        entries[variable][year] = entries[variable][year].groupby(level=variable).sum()
        pdfs[variable][year] = (entries[variable][year]/data[variable][year]).fillna(0)
        
for variable in popularity_types:
    f = data_directory+'Predictive_Models/PDF/%s/%s/'%(dummy_relatedness,variable)
    data[variable] = {}
    entries[variable] = {}
    pdfs[variable] = {}    
    for year in model_years:
        data[variable][year] = pd.read_hdf(f+'cumulative.h5', 'data/year_%i'%year).astype('float32')
        entries[variable][year] = pd.read_hdf(f+'cumulative.h5', 'entries/year_%i'%year).astype('float32')
        data[variable][year] = data[variable][year].groupby(level=variable).sum()
        entries[variable][year] = entries[variable][year].groupby(level=variable).sum()
        pdfs[variable][year] = (entries[variable][year]/data[variable][year]).fillna(0)
        

print(time()-t)


# In[11]:

from scipy.stats import rankdata

def rankify(data, data_column,
    group_size_indicator, max_group_size=629):

    n_data_points= data.shape[0]
    output = zeros(n_data_points).astype('float32')
    group_start_index = 0

    while group_start_index<n_data_points:
        group_size = max_group_size-int(data[group_size_indicator].values[group_start_index])+1
        group_stop_index = group_start_index + group_size
        output[group_start_index:group_stop_index] = rankdata(data[data_column].values[group_start_index:group_stop_index])/group_size
        group_start_index = group_stop_index
    return output


# In[12]:

from sklearn.metrics import log_loss

def loglossify(data, y_true,y_predicted,
    group_size_indicator, max_group_size=629):

    n_data_points= data.shape[0]
    output = zeros(n_data_points).astype('float32')
    group_start_index = 0

    while group_start_index<n_data_points:
        group_size = max_group_size-int(data[group_size_indicator].values[group_start_index])+1
        group_stop_index = group_start_index + group_size
        output[group_start_index:group_stop_index] = log_loss(data[y_true].values[group_start_index:group_stop_index].astype('float'),
                                                             data[y_predicted].values[group_start_index:group_stop_index].astype('float'))
        group_start_index = group_stop_index
    return output


# In[13]:

t = time()

entry_data = pd.HDFStore((data_directory+
                                 'Agent_Entries/samples/agent_entry_data_%s_sample_%i_%i_agents.h5'%(class_system,
                                                                                                         sample_start,
                                                                                                         sample_start+sample_length)))               
all_data = entry_data['all_available_classes']

all_data['Agent_Previous_Citations_to_Class'] = (all_data['Agent_Previous_Citations_to_Class']>0).astype('int')
all_data['CoAgent_Count_in_Class'] = (all_data['CoAgent_Count_in_Class']>0).astype('int')


for relatedness in relatedness_types:
    if 'z_score' in relatedness:
        if 'rescaled' in relatedness:
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
    all_data[popularity]/=100
    all_data[popularity] = digitize(all_data[popularity], arange(0,1, 1/n_bins))/n_bins
print(time()-t)


# In[1]:

for relatedness in relatedness_types:
    t = time()
    print(relatedness)
    
    for popularity in popularity_types:

        for model_year in model_years:
            column_label = '%i_NB_4D_with_%s_and_%s'%(model_year, relatedness, popularity)
            
            p = 1
            for variable in [relatedness, popularity, 
                             'Agent_Previous_Citations_to_Class',
                            'CoAgent_Count_in_Class']:
                p *= pdfs[variable][model_year][all_data[variable].values].fillna((entries[variable][model_year].sum()/
                                                                                   data[variable][model_year].sum())).values
            p = p.astype('float32')
            
            all_data['Prediction_from_'+column_label] = p#model.ix[predictors.to_records(index=False)].fillna(0).values
            all_data['Prediction_from_'+column_label] = all_data.groupby(['Agent_ID',
                                                                          'Agent_Class_Number'])['Prediction_from_'+column_label].transform(lambda x: x/sum(x))
#             all_data['Prediction_from_'+column_label] = all_data['Prediction_from_'+column_label].fillna(0)
            
            all_data['Prediction_Rank_from_'+column_label] = rankify(all_data, 'Prediction_from_'+column_label, 'Agent_Class_Number')

            all_data['Prediction_log_loss_from_'+column_label+''] = loglossify(all_data, 'Entered','Prediction_from_'+column_label,
                                                                             'Agent_Class_Number')
            del(all_data['Prediction_from_'+column_label])

    gc.collect()
    print(time()-t)


# In[16]:

for popularity in popularity_types:
    t = time()
    for model_year in model_years:
        column_label = '%i_NB_3D_with_%s'%(model_year, popularity)

        p = 1
        for variable in [popularity, 
                         'Agent_Previous_Citations_to_Class',
                        'CoAgent_Count_in_Class']:
            p *= pdfs[variable][model_year][all_data[variable].values].fillna((entries[variable][model_year].sum()/
                                                                                   data[variable][model_year].sum())).values
        p = p.astype('float32')


        all_data['Prediction_from_'+column_label] = p#model.ix[predictors.to_records(index=False)].fillna(0).values
        all_data['Prediction_from_'+column_label] = all_data.groupby(['Agent_ID',
                                                                      'Agent_Class_Number'])['Prediction_from_'+column_label].transform(lambda x: x/sum(x))
#             all_data['Prediction_from_'+column_label] = all_data['Prediction_from_'+column_label].fillna(0)

        all_data['Prediction_Rank_from_'+column_label] = rankify(all_data, 'Prediction_from_'+column_label, 'Agent_Class_Number')

        all_data['Prediction_log_loss_from_'+column_label+''] = loglossify(all_data, 'Entered','Prediction_from_'+column_label,
                                                                         'Agent_Class_Number')
        del(all_data['Prediction_from_'+column_label])

gc.collect()
print(time()-t)


# In[ ]:

entry_data['entries_with_predictions_NB_4D'] = all_data[all_data['Entered']>0]
entry_data.close()


# In[17]:

# movement_data['entries_with_predictions_NB_3D_and_4D'] = all_data[all_data['Entered']>0]
# movement_data.close()

