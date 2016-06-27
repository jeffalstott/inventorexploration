
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *
from time import time


# In[3]:

# class_system = 'IPC4'
# data_directory = '../data/'


# In[4]:

print(class_system)


# In[5]:

print("Calculating citation data") 
t = time()

citations_store = pd.HDFStore(data_directory+'citations_organized.h5')
citations = citations_store['citations']
citation_class_lookup = citations_store['%s_class_lookup'%class_system]
citations_store.close()

citation_class_lookup = citation_class_lookup.reset_index().set_index(0)
for column in citations.columns:
    if class_system in column:
        new_name = column.replace('_'+class_system, "")
        citations.rename(columns={column: new_name}, inplace=True)


# In[21]:

store = pd.HDFStore(data_directory+'patent_class_citation_count.h5')
if class_system not in store:
    patent_class_citation_count = citations.groupby('Citing_Patent')['Class_Cited_Patent'].value_counts()
    store[class_system] = patent_class_citation_count
    store['class_lookup_table_%s'%class_system] = citation_class_lookup
    print(time()-t)
store.close()


# In[41]:

store = pd.HDFStore(data_directory+'class_citation_counts.h5')
if class_system not in store:
    a = set(citations['Class_Citing_Patent'].dropna().unique())
    b = set(citations['Class_Cited_Patent'].dropna().unique())
    all_classes = sort(list(a.union(b)))
    all_years = sort(citations['Year_Citing_Patent'].unique())
    citation_counts = pd.DataFrame(index=pd.MultiIndex.from_product((all_classes, all_years),
                                                                   names=['Class_ID', 'Issued_Year']))

    for citation_type, class_column in [('Outward', 'Class_Citing_Patent'),
                                        ('Inward', 'Class_Cited_Patent')]:    
        count_by_year = citations.groupby([class_column, 'Year_Citing_Patent'])['Citing_Patent'].count().sort_index()
        cumulative_count_by_year = count_by_year.groupby(level=class_column).cumsum()
        citation_counts['Class_%s_Citation_Count'%citation_type] = count_by_year
        citation_counts['Class_Cumulative_%s_Citation_Count'%citation_type] = cumulative_count_by_year
    store[class_system] = citation_counts.reset_index()
    store['class_lookup_table_%s'%class_system] = citation_class_lookup
store.close()


# In[ ]:

# class_citations_dict = {}

# for citation_type, class_column in [('Outward', 'Class_Citing_Patent'),
#                                     ('Inward', 'Class_Cited_Patent')]:
#     count_by_year = citations.groupby([class_column, 'Year_Citing_Patent'])['Citing_Patent'].count().sort_index()
#     cumulative_count_by_year = count_by_year.groupby(level=class_column).cumsum().reset_index()
#     count_by_year = count_by_year.reset_index()

#     class_citations_dict['Class_%s_Citation_Count'%citation_type] = count_by_year.reset_index().rename(columns={class_column:'Class_ID',
#                                                                                                                                 'Citing_Patent': 'Count'})
#     class_citations_dict['Class_Cumulative_%s_Citation_Count'%citation_type] = cumulative_count_by_year.rename(columns={0:'Count',
#                                                                                                                                          class_column:'Class_ID'})
# for k in class_citations_dict.keys():
#     class_citations_dict[k].rename(columns={'Year_Citing_Patent': 'Issued_Year'}, inplace=True)

#     #The stored citations data may have a different class_lookup index than we have calculated here
#     #so we convert it to ours.
#     class_citations_dict[k]['Class_ID'] = classes_lookup.set_index('Class_Name').ix[citation_class_lookup.ix[
#             class_citations_dict[k]['Class_ID']]['index']]['Class_ID'].values


# In[ ]:

store = pd.HDFStore(data_directory+'patent_class_citation_count.h5')
if class_system not in store:
    store[class_system] = patent_class_citation_count
    store['class_lookup_table_%s'%class_system] = citation_class_lookup
    print(time()-t)
store.close()


# In[7]:

print("Calculating agent-patent relationships")
t = time()


# In[8]:

all_inventorships = pd.read_csv(data_directory+'disamb_data_ipc_citations_2.csv')

if class_system=='IPC':
    temp_class_system = 'IPC3'
else:
    temp_class_system = class_system
    
agent_column = 'INVENTOR_ID'

all_data = all_inventorships[['PID', agent_column, temp_class_system, 
                              'APPDATE', 'GYEAR',
                             'CITED_CNTS']]

all_data.rename(columns={'PID': 'Patent',
                     agent_column: 'Agent',
                     temp_class_system: 'Class',
                     'APPDATE': 'Application_Date',
                     'GYEAR': 'Issued_Year',
                        'CITED_CNTS': 'Citations'},
            inplace=True)

all_data.drop_duplicates(inplace=True)

del(all_inventorships)

all_classes_observed = sort(all_data.Class.unique())
n_classes = len(all_classes_observed)

classes_lookup = pd.DataFrame(data=all_classes_observed, 
                              columns=['Class_Name'])
classes_lookup['Class_ID'] = classes_lookup.index

all_data['Class_ID'] = classes_lookup.set_index('Class_Name').ix[all_data['Class'],'Class_ID'].values

all_data.drop(['Class'], axis=1, inplace=True)

all_data['Application_Year'] = pd.DatetimeIndex(all_data['Application_Date']).year


# In[1]:

store = pd.HDFStore(data_directory+'agent_patent_relationships.h5')

if 'agent_patent_lists' not in store:
    agent_patent_lists = all_data.groupby('Agent')['Patent'].apply(lambda x: list(x))
    store.put('/agent_patent_lists', agent_patent_lists)
    
if 'agent_patent_year_lists' not in store:
    agent_patent_year_lists = all_data.groupby('Agent')['Issued_Year'].apply(lambda x: list(x))
    store.put('/agent_patent_year_lists', agent_patent_year_lists)
    
if 'patent_agent_lists' not in store:
    patent_agent_lists = all_data.groupby('Patent')['Agent'].apply(lambda x: list(x))
    store.put('/patent_agent_lists', patent_agent_lists)

if 'patent_classes_%s'%class_system not in store:
    patent_classes = all_data.drop_duplicates('Patent')[['Patent', 'Application_Year', 'Class_ID']].set_index('Patent')
    store.put('/patent_classes_%s'%class_system, patent_classes)
    
if 'class_lookup_table_%s'%class_system not in store:
    store.put('/class_lookup_table_%s'%class_system, classes_lookup)
    
store.close()
print(time()-t)

