
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *


# In[2]:

# class_system = 'IPC4'
# data_directory = '../data/'


# In[3]:

# target = 'entries'
# target = 'entries_with_predictions'
# target = 'entries_with_predictions_NB_3D_and_4D'
# target = 'entries_with_predictions_3D_Popularity_Citations_CoAuthors'


# In[4]:

store = pd.HDFStore(data_directory+'Agent_Entries/agent_%s_%s.h5'%(target, class_system))
entries = store['%s_%s'%(target, class_system)]
store.close()


# In[5]:

entries['Application_Year'] = entries['Application_Year'].astype('int')
entries['Class_ID'] = entries['Class_ID'].astype('int')


# In[6]:

store = pd.HDFStore(data_directory+'organized_patent_data.h5')
agent_lookup = store['agents_lookup_explorers_%s'%class_system]
patent_data = store['data_%s'%class_system]
store.close()


# In[7]:

entries['Agent'] = agent_lookup.ix[entries['Agent_ID'], 'Disambiguation_ID'].values


# In[8]:

patent_performance = patent_data[['Patent', 'Citations', 'Application_Year', 'Class_ID']].drop_duplicates('Patent').set_index('Patent')


# In[9]:

patent_performance['Citations_Percentile_for_Year'] = patent_performance.groupby('Application_Year')['Citations'].apply(lambda x: x.rank(method='min', pct=True))
patent_performance['Citations_Percentile_for_Year_and_Class'] = patent_performance.groupby(['Application_Year', 'Class_ID'])['Citations'].apply(lambda x: x.rank(method='min', pct=True))

patent_performance['Citations_Hit99_for_Year'] = (patent_performance['Citations_Percentile_for_Year']>.99).astype('int')
patent_performance['Citations_Hit99_for_Year_and_Class'] = (patent_performance['Citations_Percentile_for_Year_and_Class']>.99).astype('int')

patent_data['Citations_Percentile_for_Year'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Percentile_for_Year'].values
patent_data['Citations_Percentile_for_Year_and_Class'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Percentile_for_Year_and_Class'].values

patent_data['Citations_Hit99_for_Year'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Hit99_for_Year'].values
patent_data['Citations_Hit99_for_Year_and_Class'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Hit99_for_Year_and_Class'].values


# In[10]:

zscore = lambda x: (x - x.mean()) / x.std()
    
patent_performance['Citations_Z_for_Year'] = patent_performance.groupby('Application_Year')['Citations'].transform(zscore)
patent_performance['Citations_Z_for_Year_and_Class'] = patent_performance.groupby(['Application_Year', 'Class_ID'])['Citations'].transform(zscore)


# In[11]:

patent_data['Citations_Z_for_Year'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Z_for_Year'].values
patent_data['Citations_Z_for_Year_and_Class'] = patent_performance.ix[patent_data['Patent'].values,  'Citations_Z_for_Year_and_Class'].values


# In[12]:

thresholds = [3,4,5,6]
for thr in thresholds:
    patent_data['Citations_%iZ_for_Year_and_Class'%thr] = patent_data['Citations_Z_for_Year_and_Class']>thr
    patent_data.groupby(['Agent', 'Class_ID'])['Citations_%iZ_for_Year_and_Class'%thr].sum()
    entries['Citations_Hit_for_Year_and_Class_%iZ_Count'%thr] = patent_data.groupby(['Agent', 
        'Class_ID'])['Citations_%iZ_for_Year_and_Class'%thr].sum().ix[pd.Index(entries[['Agent',
                                                                                        'Class_ID']])].values
    entries['Citations_Hit_for_Year_and_Class_%iZ_Rate'%thr] = (entries['Citations_Hit_for_Year_and_Class_%iZ_Count'%thr]/
                                                                entries['Agent_Number_of_Patents_in_Class']
                                                                )


# In[13]:

entries['Highest_Patent_Citations_in_Class'] = patent_data.groupby(['Agent', 
                                                             'Class_ID'])['Citations'].max().ix[pd.Index(entries[['Agent', 
                                                                                                            'Class_ID']])].values
entries['Highest_Patent_Citations_Percentile_for_Year_in_Class'] = patent_data.groupby(['Agent', 
                                                             'Class_ID'])['Citations_Percentile_for_Year'].max().ix[pd.Index(entries[['Agent', 
                                                                                                            'Class_ID']])].values
entries['Highest_Patent_Citations_Percentile_for_Year_and_Class_in_Class'] = patent_data.groupby(['Agent', 
                                                             'Class_ID'])['Citations_Percentile_for_Year_and_Class'].max().ix[pd.Index(entries[['Agent', 
                                                                                                            'Class_ID']])].values

entries['Citations_Hit99_for_Year_Rate_in_Class'] = patent_data.groupby(['Agent', 
                                                             'Class_ID'])['Citations_Hit99_for_Year'].mean().ix[pd.Index(entries[['Agent', 
                                                                                                                      'Class_ID']])].values
entries['Citations_Hit99_for_Year_and_Class_Rate_in_Class'] = patent_data.groupby(['Agent', 
                                                             'Class_ID'])['Citations_Hit99_for_Year_and_Class'].mean().ix[pd.Index(entries[['Agent', 
                                                                                                                      'Class_ID']])].values


# In[14]:

entries['First_Patent_Citations'] = entries['Citations']


# In[15]:

entries['First_Patent_Citations_Percentile_for_Year'] = patent_performance.ix[entries['Patent'].values, 'Citations_Percentile_for_Year'].values
entries['First_Patent_Citations_Percentile_for_Year_and_Class'] = patent_performance.ix[entries['Patent'].values, 'Citations_Percentile_for_Year_and_Class'].values

entries['First_Patent_Citations_Hit99_for_Year'] = patent_performance.ix[entries['Patent'].values, 'Citations_Hit99_for_Year'].values
entries['First_Patent_Citations_Hit99_for_Year_and_Class'] = patent_performance.ix[entries['Patent'].values, 'Citations_Hit99_for_Year_and_Class'].values


# In[16]:

values_to_calculate = [('First_Patent_Citations','Citations')
                       ]
    
for entries_column, patent_data_column in values_to_calculate:
    print(entries_column)
    m = patent_performance.groupby('Application_Year')[patent_data_column].mean()
    entries[entries_column+'_Mean_for_Year'] = m.ix[entries['Application_Year']].values
    
    s = patent_performance.groupby('Application_Year')[patent_data_column].std()
    entries[entries_column+'_STD_for_Year'] = s.ix[entries['Application_Year']].values
    
    m = patent_performance.groupby(['Application_Year', 'Class_ID'])[patent_data_column].mean()
    entries[entries_column+'_Mean_for_Year_and_Class'] = m.ix[zip(entries['Application_Year'], 
                                                                  entries['Class_ID'])].values
    
    s = patent_performance.groupby(['Application_Year', 'Class_ID'])[patent_data_column].std()
    entries[entries_column+'_STD_for_Year_and_Class'] = s.ix[zip(entries['Application_Year'], 
                                                                 entries['Class_ID'])].values


# In[17]:

# patent_data.sort(['Agent', 'Application_Date', 'Patent'], inplace=True)


# In[18]:

patent_data.sort(['Application_Date', 'Patent'], inplace=True)

patent_data['Agent_Number_of_Patents_in_Class_All_Time'] = patent_data.groupby(['Agent', 'Class_ID'])['Patent'].transform('count')
patent_data['Agent_Number_of_Previous_Patents_in_Class'] = patent_data.groupby(['Agent', 'Class_ID'])['Patent'].cumcount()
patent_data['Agent_Number_of_Further_Patents_in_Class'] = patent_data['Agent_Number_of_Patents_in_Class_All_Time'] - patent_data['Agent_Number_of_Previous_Patents_in_Class']

patent_data['Agent_Number_of_Previous_Patents'] = patent_data.groupby(['Agent'])['Patent'].cumcount()
patent_data['Agent_Number_of_Further_Patents'] = patent_data['Agent_Number_of_Patents_All_Time'] - patent_data['Agent_Number_of_Previous_Patents']


# In[19]:

patent_data.sort(['Application_Date', 'Patent'], ascending=False, inplace=True)

patent_data['Agent_Number_of_Citations_from_Further_Patents_in_Class'] = patent_data.groupby(['Agent', 'Class_ID'])['Citations'].cumsum()

patent_data['Agent_Number_of_Citations_from_Further_Patents'] = patent_data.groupby(['Agent'])['Citations'].cumsum()


# In[20]:

entries['Agent_Number_of_Citations_per_Patent_in_Class'] = (entries['Agent_Number_of_Citations_in_Class'] /
                                                            entries['Agent_Number_of_Patents_in_Class'])


# In[21]:

patent_data['Agent_Number_of_Citations_per_Patent_from_Further_Patents_in_Class'] = (patent_data['Agent_Number_of_Citations_from_Further_Patents_in_Class'] /
                                                                                     patent_data['Agent_Number_of_Further_Patents_in_Class']
                                                                                     )


# In[22]:

from scipy.stats import scoreatpercentile


# In[23]:

values_to_calculate = [('Agent_Number_of_Patents_in_Class','Agent_Number_of_Further_Patents_in_Class'),
                       ('Agent_Number_of_Citations_in_Class','Agent_Number_of_Citations_from_Further_Patents_in_Class'),
                       ('Agent_Number_of_Citations_per_Patent_in_Class', 'Agent_Number_of_Citations_per_Patent_from_Further_Patents_in_Class')
                       ]

new_to_class = patent_data['Agent_Number_of_Previous_Patents_in_Class']==0
newborn = patent_data['Agent_Patent_Number']==1
native = patent_data['Agent_Class_Number']==1
for these_patent_data, label in [(None, '_of_All_Agents_Active_in_Class'),
                                (new_to_class, '_of_Agents_New_to_Class'),
                                (newborn, '_of_Agents_Newborn_in_Class'),
                                (new_to_class*~newborn, '_of_New_Immigrants_to_Class'),
                                (~new_to_class, '_of_Agents_Previously_Active_in_Class'),
                                (~new_to_class*native, '_of_Natives_Previously_Active_in_Class')
                                ]:
    print(label)
    if these_patent_data is not None:
        these_patent_data = patent_data[these_patent_data]
    else:
        these_patent_data = patent_data
        
    for entries_column, patent_data_column in values_to_calculate:
        print(entries_column)
        grouper = these_patent_data.drop_duplicates(['Agent', 'Application_Year', 'Class_ID']).groupby('Application_Year')[patent_data_column]
        m = grouper.mean()
        s = grouper.std()
        h99 = grouper.apply(lambda x: scoreatpercentile(x,99))
        h90 = grouper.apply(lambda x: scoreatpercentile(x,90))
        
        entries[entries_column+'_Mean_for_Year'+label] = m.ix[entries['Application_Year']].values
        entries[entries_column+'_STD_for_Year'+label] = s.ix[entries['Application_Year']].values
        entries[entries_column+'_Percentile_99_for_Year'+label] = h99.ix[entries['Application_Year']].values
        entries[entries_column+'_Percentile_90_for_Year'+label] = h90.ix[entries['Application_Year']].values


        grouper = these_patent_data.drop_duplicates(['Agent', 'Application_Year', 'Class_ID']).groupby(['Application_Year', 'Class_ID'])[patent_data_column]
        m = grouper.mean()
        s = grouper.std()
        h99 = grouper.apply(lambda x: scoreatpercentile(x,99))
        h90 = grouper.apply(lambda x: scoreatpercentile(x,90))
        
        entries[entries_column+'_Mean_for_Year_and_Class'+label] = m.ix[zip(entries['Application_Year'], entries['Class_ID'])].values
        entries[entries_column+'_STD_for_Year_and_Class'+label] = s.ix[zip(entries['Application_Year'], entries['Class_ID'])].values
        entries[entries_column+'_Percentile_99_for_Year_and_Class'+label] = h99.ix[zip(entries['Application_Year'], entries['Class_ID'])].values
        entries[entries_column+'_Percentile_90_for_Year_and_Class'+label] = h90.ix[zip(entries['Application_Year'], entries['Class_ID'])].values


# In[24]:

entries['First_Patent_Application_Year']= patent_data[patent_data['Agent_Patent_Number']==1].set_index('Agent').ix[entries['Agent'], 
                                                                                                                   'Application_Year'].values
entries['Years_Since_First_Patent'] = entries['Application_Year'] - entries['First_Patent_Application_Year']


# In[25]:

store = pd.HDFStore(data_directory+'Agent_Entries/agent_%s_%s.h5'%(target, class_system))
store['%s_%s_with_performance'%(target, class_system)] = entries
store.close()

