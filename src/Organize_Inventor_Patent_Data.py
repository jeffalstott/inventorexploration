
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *


# In[2]:

# data_directory = '../data/'
# class_system = 'IPC4'


# In[3]:

all_inventorships = pd.read_csv(data_directory+'disamb_data_ipc_citations_2.csv')


# In[4]:

agent_column = 'INVENTOR_ID'


if class_system == 'IPC':
    temp_class_system = 'IPC3'
else:
    temp_class_system = class_system


data = all_inventorships[['PID', agent_column, temp_class_system, 
                          'APPDATE', 'GYEAR',
                         'CITED_CNTS']]

data.rename(columns={'PID': 'Patent',
                     agent_column: 'Agent',
                     temp_class_system: 'Class',
                     'APPDATE': 'Application_Date',
                     'GYEAR': 'Issued_Year',
                    'CITED_CNTS': 'Citations'},
            inplace=True)

data.drop_duplicates(inplace=True)

data['Application_Date'] = pd.to_datetime(data['Application_Date'])
data['Application_Year'] = pd.DatetimeIndex(data['Application_Date']).year


# In[5]:

official_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', '%s_class_lookup'%class_system)

all_classes_observed = sort(data.Class.unique())
classes_lookup = pd.DataFrame(data=official_class_lookup.ix[all_classes_observed].dropna().index.values,#all_classes_observed, 
                              columns=['Class_Name'])
classes_lookup['Class_ID'] = classes_lookup.index

n_classes = classes_lookup.shape[0]

data['Class_ID'] = classes_lookup.set_index('Class_Name').ix[data['Class'],'Class_ID'].values
data.dropna(subset=['Class_ID'], inplace=True)
data['Class_ID'] = data['Class_ID'].astype('int32')
data.drop('Class', axis=1, inplace=True)


# In[9]:

Agent_patent_counts = data['Agent'].value_counts()
Agent_class_counts = data.drop_duplicates(['Agent', 'Class_ID'])['Agent'].value_counts()

agents_lookup = pd.DataFrame({"Agent_Number_of_Patents_All_Time": Agent_patent_counts, 
              "Agent_Number_of_Classes_All_Time": Agent_class_counts})

data = data.merge(agents_lookup, left_on='Agent', right_index=True, how='inner')

data.sort(['Agent', 'Application_Date', 'Patent'], inplace=True)
data['Agent_Patent_Number'] = data.groupby('Agent')['Patent'].cumcount()+1

data['New_Class'] = data.groupby('Agent')['Class_ID'].transform(lambda x: ~x.duplicated())

def f(classes):
    sorted_unique, inverse_unique, indices = unique(classes, return_inverse=True, return_index=True)
    z, order_of_appearance = unique(inverse_unique, return_inverse=True)
    return order_of_appearance[indices]
data['Agent_Class_Number'] = data.groupby('Agent')['Class_ID'].transform(f)
data['Agent_Class_Number'] += 1

data['Agent_Class_Number'] = data['Agent_Class_Number'].astype('int')
data['New_Class'] = data['New_Class'].astype('uint8')


# In[10]:

agents_lookup_explorers = agents_lookup[agents_lookup.Agent_Number_of_Classes_All_Time>=2]
agents_lookup_explorers['Agent_ID'] = arange(len(agents_lookup_explorers.index))

agents_lookup_explorers.index.name = 'Disambiguation_ID'
agents_lookup_explorers.reset_index(inplace=True)
agents_lookup_explorers.set_index(['Agent_ID'], inplace=True)


# In[11]:

store = pd.HDFStore(data_directory+'organized_patent_data.h5')
store['data_%s'%class_system] = data
store['classes_lookup_%s'%class_system] = classes_lookup
store['agents_lookup_%s'%class_system] = agents_lookup
store['agents_lookup_explorers_%s'%class_system] = agents_lookup_explorers
store.close()

