
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *
# data_directory = '../data/'
# class_system = 'IPC4'
# sample_length = 1000 


# In[2]:

# target = 'entries'
# target = 'entries_with_predictions_NB_4D_rescaled'
# target = 'entries_with_predictions'
# target = 'entries_with_predictions_NB_4D'
# target = 'entries_with_predictions_3D_Popularity_Citations_CoAuthors'


# In[5]:

n_rows = 0
this_sample = 0
while True:
    print(this_sample)
    try:
        this_data = pd.HDFStore(data_directory+"Agent_Entries/samples/"
                                "agent_entry_data_{2}_sample_{0}_{1}_agents.h5".format(this_sample,
                                                                                      this_sample+sample_length,
                                                                                         class_system),
                               mode='r')
    except:
        break
    n_rows += this_data.get_storer(target).shape[0]
    this_sample += sample_length
    this_data.close()


# In[6]:

this_sample = 0
all_data = None
this_row = 0
while True:
    print(this_sample)
    try:
        this_data = pd.HDFStore(data_directory+"Agent_Entries/samples/"
                                "agent_entry_data_{2}_sample_{0}_{1}_agents.h5".format(this_sample,
                                                                                      this_sample+sample_length,
                                                                                         class_system),
                               mode='r')
    except:
        break
    n_rows_this_data = this_data.get_storer(target).shape[0]

    if all_data is None:
        import gc
        gc.collect()
        print('Initializing DataFrame with %i rows'%(n_rows))
        all_data = pd.DataFrame(index=range(n_rows),
                                columns=this_data[target].columns,
                               dtype='float32')

    all_data.iloc[this_row:this_row+n_rows_this_data] = this_data[target].values.astype('float32')
    this_row += n_rows_this_data
    this_sample += sample_length
    this_data.close()


# In[52]:

store = pd.HDFStore(data_directory+'Agent_Entries/agent_%s_%s.h5'%(target, class_system))
store.put('/%s_%s'%(target, class_system), all_data, append=False)
store.close()

