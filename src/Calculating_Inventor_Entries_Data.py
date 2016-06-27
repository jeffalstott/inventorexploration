
# coding: utf-8

# Set Up
# ====

# In[1]:

import pandas as pd
import seaborn as sns
from pylab import *
import gc


# In[2]:

# use_precalculated_supporting_data = True
# store_calculated_supporting_data = False
# use_regressed_z_scores = True
# data_directory = '../data/'
# all_n_years = ['all', 1, 5]
# class_system = 'IPC4'
# agent_sample = (0,1000)
# # agent_sample = None


# In[3]:

def create_n_years_label(n_years, suffix=False):
    if n_years is None or n_years=='all' or n_years=='cumulative':
        n_years_label = ''
        return n_years_label
    else:
        n_years_label = '%i_years'%n_years
    if suffix:
        n_years_label = '_'+n_years_label
    else:
        n_years_label = n_years_label+'_'
    return n_years_label


# Pull in Data
# ===

# In[4]:

store = pd.HDFStore(data_directory+'organized_patent_data.h5')

all_data = store['data_%s'%class_system]
classes_lookup = store['classes_lookup_%s'%class_system]
agents_lookup = store['agents_lookup_explorers_%s'%class_system]
store.close()


# In[5]:

n_classes = classes_lookup.shape[0]


# Take Sample
# ===

# In[14]:

if type(agent_sample)==int:
    from numpy.random import choice
    sampled_agents = agents_lookup.iloc[choice(agents_lookup.shape[0], agent_sample, replace=False)].reset_index()[['Agent_ID', 'Disambiguation_ID']]
elif agent_sample is not None:
    sampled_agents = agents_lookup.iloc[agent_sample[0]:agent_sample[1]].reset_index()[['Agent_ID', 'Disambiguation_ID']]

if agent_sample is not None:
    data = all_data.merge(sampled_agents, left_on='Agent', right_on='Disambiguation_ID')

print("%i unique agents after sampling"%len(data.Agent_ID.unique()))
print("%i inventorships after sampling"%data.shape[0])


# In[15]:

if use_precalculated_supporting_data:
    del(all_data)
del(data['Disambiguation_ID'])
del(data['Agent'])


# In[16]:

agent_home_classes = data.groupby('Agent_ID')['Class_ID'].first()
data['Agent_Home_Class'] = agent_home_classes.ix[data['Agent_ID']].values


# Calculate Agents' Performance in Class
# ===
# (Number of Patents and Citations in Class)

# In[17]:

data['Agent_Number_of_Patents_in_Class'] = data.groupby(['Agent_ID', 'Class_ID'])['Patent'].transform('count')
data['Agent_Number_of_Citations_in_Class'] = data.groupby(['Agent_ID', 'Class_ID'])['Citations'].transform('sum')


# Calculate Agent Patent Diversity Data
# ===

# In[18]:

def expanding_diversity(x):
    x = x.values
    count_dict = {}
    diversities = zeros((len(x),2))
    for i in range(len(x)-1):
        X = x[i]
        try:
            count_dict[X] +=1
        except:
            count_dict[X] = 1
            
        cum_counts = array(list(count_dict.values()))
        shares = cum_counts/sum(cum_counts)
        
        diversities[i+1,0] = sum(shares**2) #First column is Herfindal
        diversities[i+1,1] = sum(shares*-log(shares)) #Second column is entropy
        #We leave the first row as 0. We are measuring how much diversity there was immediately BEFORE this step.
    return pd.DataFrame(diversities)

z = data.groupby(['Agent_ID'])['Class_ID'].apply(expanding_diversity)
data['Agent_Class_Diversity_Herfindahl_Index'] = z[0].values.astype('float32')
data['Agent_Class_Diversity_Entropy'] = z[1].values.astype('float32')
del(z)


# Identify when agents enter into new classes
# ===

# In[19]:

new_class_entries = data.ix[(data['New_Class']>0)]#*(data['Agent_Class_Number']>1)]


# Expand dataframe to include the classes the agents didn't enter
# ====

# In[20]:

### Mark which classes were actually entered
new_class_data = new_class_entries.copy()
new_class_data['Entered'] = True
new_class_data.reset_index(inplace=True)


# For every new entry, create a list of every class that COULD have been entered
# I.e. all the unentered classes at that point

# In[21]:

### Define the function that identifies and spits out the classes that could have been entered,
### creating the new index for the data frame
def multiindex_maker(input_line,input_data, 
                     output_line, output_data,
                     function_to_apply=None,
                     function_data=None):
    
    agent = input_data[input_line,0]
    entry_number = input_data[input_line,1]
    if entry_number == 1: 
        ### Skip the first class entry, as that's not what we're modeling.
        ### Later on, not having these rows in the multiindex will remove these starting points
        ### From the data to be analyzed
        return output_line

    classes_previously_entered = input_data[1+input_line-entry_number:input_line,2] #Agent_Class_Number is 1-indexed
    classes_available = sort(list(set(range(n_classes)).difference(classes_previously_entered)))
    n_reps = len(classes_available)
    
    output_data[output_line:output_line+n_reps,0] = agent
    output_data[output_line:output_line+n_reps,1] = entry_number
    output_data[output_line:output_line+n_reps,2] = classes_available
    
    if function_to_apply:
        function_to_apply(input_line,input_data, 
                          output_line, output_data,
                          classes_previously_entered,
                          classes_available,
                          n_reps,
                          function_data)
    return output_line + n_reps


# In[22]:

n_rows_to_be_created = sum(new_class_entries.Agent_ID.value_counts().map(lambda n: sum(n_classes-1-arange(n-1))))

input_data = new_class_entries.sort(['Agent_ID', 
                                     'Agent_Class_Number'])[['Agent_ID', 
                                                             'Agent_Class_Number', 
                                                             'Class_ID']].values
output_data = zeros((n_rows_to_be_created, 3))

output_line = 0
for input_line in arange(len(input_data)):
    output_line = multiindex_maker(input_line, input_data, output_line, output_data)


# In[23]:

gc.collect()


# In[24]:

### We now have a set of every agent, their class entry numbers, and the classes they
### could have entered at that time. We make this into a multiindex, and reindex the
### new class data to use that index. All the blank spots (i.e. the unentered classes)
### will be added to the data frame as empty rows (nans)
multiindex = output_data

multiindex = pd.MultiIndex.from_arrays([multiindex[:,0],
                                        multiindex[:,1],
                                        multiindex[:,2]], 
              names=['Agent_ID','Agent_Class_Number','Class_ID'])

new_class_data.set_index(['Agent_ID', 'Agent_Class_Number', 'Class_ID'], inplace=True)
new_class_data = new_class_data.reindex(multiindex)


# In[25]:

gc.collect()


# In[26]:

for c in new_class_data.columns:
    if new_class_data[c].dtype == 'float64':
        new_class_data[c] = new_class_data[c].astype('float32')


# In[27]:

gc.collect()


# In[28]:

new_class_data.reset_index(inplace=True)
new_class_data.drop(['index'], axis=1, inplace=True)


# In[29]:

gc.collect()


# Fill in missing data

# In[30]:

for col in ['Application_Year', 'Issued_Year', 'Application_Date', 
            'Agent_Class_Diversity_Herfindahl_Index', 'Agent_Class_Diversity_Entropy',
            'Agent_Number_of_Patents_in_Class', 'Agent_Number_of_Citations_in_Class',
            'Agent_Number_of_Classes_All_Time', 'Agent_Number_of_Patents_All_Time',
            'Agent_Patent_Number'
           ]:
    new_class_data[col] = new_class_data.groupby(['Agent_ID', 
                                                 'Agent_Class_Number'])[col].transform(
                                                    lambda x: x.fillna(
                                                        x[x.first_valid_index()]
                                                    )
        )


# In[31]:

new_class_data['Entered'].fillna(False, inplace=True)
new_class_data['Entered'] = new_class_data['Entered'].astype('uint8')


# In[32]:

for c in new_class_data.columns:
    if new_class_data[c].dtype == 'float64':
        new_class_data[c] = new_class_data[c].astype('float32')
gc.collect()


# Make simple predictors
# ====

# In[36]:

### Code to rank predictors, once we calculate them.
def rankify(column_name,data=new_class_data):
    data[column_name+'_Percentile'] = (100. * (data.groupby(['Agent_ID','Agent_Class_Number'])[column_name].rank() / 
                                              data.groupby(['Agent_ID','Agent_Class_Number'])[column_name].transform('count')
                                              )).astype('float32')
    data.drop(column_name, axis=1, inplace=True)
    return


# Class' popularity up until the inventor entered them

# In[37]:

# class_counts_dict = {}
# class_by_year_multiindex = pd.MultiIndex.from_product((classes_lookup['Class_ID'], 
#                                                        range(all_data.Issued_Year.min(), 
#                                                              all_data.Issued_Year.max()+1)),
#                                                       names=('Class_ID', 'Issued_Year'))


# In[31]:

# ### Patent count
# class_counts_dict['Class_Patent_Count'] = all_data.groupby(['Class_ID', 'Issued_Year'])['Patent'].nunique()
# class_counts_dict['Class_Patent_Count'] = class_counts_dict['Class_Patent_Count'].reindex(class_by_year_multiindex).fillna(0)

# class_counts_dict['Class_Patent_Count_Cumulative'] = class_counts_dict['Class_Patent_Count'].groupby(level='Class_ID').cumsum()
# class_counts_dict['Class_Patent_Count_Cumulative'].index.name = 'Class_ID'


# # class_counts_dict['Class_Patent_Count'] class_size.groupby(level='Class').apply(lambda x: pd.rolling_sum(x, n_years))


# In[32]:

# ### Agent count per year (repeats agents from year to year)
# class_counts_dict['Class_Agent_Count'] = all_data.groupby(['Class_ID', 'Issued_Year'])['Agent'].nunique()
# class_counts_dict['Class_Agent_Count'] = class_counts_dict['Class_Agent_Count'].reindex(class_by_year_multiindex).fillna(0)

# ### Agent new count by year (no repeats of agents from year to year)
# class_counts_dict['Class_New_Agent_Count'] = new_class_entries.groupby(['Class_ID', 'Issued_Year'])['Agent_ID'].count()
# class_counts_dict['Class_New_Agent_Count'] = class_counts_dict['Class_New_Agent_Count'].reindex(class_by_year_multiindex).fillna(0)


# ### Agent cumulative count by year (no repeats of agents from year to year)
# class_counts_dict['Class_New_Agent_Count_Cumulative'] = class_counts_dict['Class_New_Agent_Count'].groupby(level='Class_ID').cumsum()
# class_counts_dict['Class_New_Agent_Count_Cumulative'].index.name = 'Class_ID'


# In[38]:

class_counts_dict = {}

counts_data = pd.HDFStore(data_directory+'popularity_counts.h5')
for n_years in all_n_years:
    print(n_years)
    n_years_label = create_n_years_label(n_years)
    n_years_label_suffix = create_n_years_label(n_years,suffix=True)
    class_counts_dict['Class_Patent_Count%s'%n_years_label_suffix] = counts_data['patent_count_%s%s'%(n_years_label, 
                                                                                   class_system)]
    class_counts_dict['Class_New_Inventor_Count%s'%n_years_label_suffix] = counts_data['new_inventor_count_%s%s'%(n_years_label, 
                                                                                   class_system)]

counts_data.close()


# In[39]:

for k in class_counts_dict.keys():
    class_counts_dict[k] = class_counts_dict[k].reset_index()
    class_counts_dict[k].rename(columns={'Agent_ID': 'Count',
                                         'Agent': 'Count',
                                         'Patent': 'Count',
                                         'patent': 'Count',
                                         'inventor': 'Count',
                                         'Year': 'Issued_Year',
                                         0: 'Count'
                                         }, inplace=True)

    #Imported data will use class names, which should be converted to numeric class ids using our lookup table
    if 'Class' in class_counts_dict[k].columns and 'Class_ID' not in class_counts_dict[k].columns:
        class_counts_dict[k]['Class_ID'] = classes_lookup.set_index('Class_Name').ix[class_counts_dict[k]['Class']].values
    


# In[40]:

for k in class_counts_dict.keys():
    class_counts_dict[k]['Application_Year'] = class_counts_dict[k]['Issued_Year']+1 
    #We're calling this "Application_Year", but really it's the application year it will be paired with.
    #It's the application year that will have this information present. (Note the **+1**)
    new_class_data[k+'_Previous_Year'] = new_class_data[['Class_ID','Application_Year']].merge(class_counts_dict[k],
                                      on=['Class_ID', 'Application_Year'],
                                      how='left')['Count'].fillna(0)
    rankify(k+'_Previous_Year',data=new_class_data)


# Make network predictors
# ====

# In[41]:

def calculate_connectivities(connectivity_metric,
#                              connectivity_metric_prefix='',
                             connectivity_metric_suffix='',
                             networks=None,
                             new_class_data=new_class_data,
                             new_class_entries=new_class_entries,
                             use_the_future=False,
                             combination_method='mean',
                             additional_label=''
                             ):

    n_rows_to_be_created = new_class_data.shape[0]

    input_data = new_class_entries.sort(['Agent_ID', 
                                         'Agent_Class_Number'])[['Agent_ID', 
                                                                 'Agent_Class_Number', 
                                                                 'Class_ID', 
                                                                 'Application_Year']].values
    if use_the_future:
       input_data[:,3] = 0
    else:
        year_lookup = pd.DataFrame(index=networks.items,
                                   columns=['Year_ID'],
                                   data=arange(len(networks.items))
                                   )
        input_data[:,3] = year_lookup.ix[input_data[:,3]-1, 'Year_ID']
    output_data = zeros((n_rows_to_be_created, 4))
    
    network_arrays = networks.ix[connectivity_metric].values

    output_line = 0
    for input_line in arange(len(input_data)):
        output_line = multiindex_maker(input_line, input_data, 
                                       output_line, output_data,
                                       function_to_apply= lambda *args: calculate_connectivities_helper(*args, 
                                                                                                       combination_method=combination_method),
                                       function_data=network_arrays)

    new_class_data.sort(['Agent_ID', 'Agent_Class_Number'], inplace=True)
    new_class_data[connectivity_metric+connectivity_metric_suffix+'_'+additional_label+'_'+combination_method] = output_data[:,3]
    return

def calculate_connectivities_helper(input_line, input_data, 
                                    output_line, output_data,
                                    classes_previously_entered,
                                    classes_available,
                                    n_reps,
                                    network_arrays,
                                    combination_method):

    year_index = input_data[input_line,3]
    if isnan(year_index) or year_index==-9223372036854775808: #If we asked for a year that we don't have network information on
        connectivity_from_previous = 0 #Sorry, can't help you.
    else:
        connectivity_from_previous = network_arrays[year_index][classes_previously_entered][:,classes_available]
        if combination_method=='mean':
            connectivity_from_previous = connectivity_from_previous.mean(axis=0) 
        if combination_method=='max':
            connectivity_from_previous = connectivity_from_previous.max(axis=0)  

    output_data[output_line:output_line+n_reps,3] = connectivity_from_previous 
    return    


# In[42]:

store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5',mode='r')


# In[43]:

summary_statistic_label='z_score'

if use_regressed_z_scores:
    regression_label = 'regressed_'
else:
    regression_label = ''

for n_years in all_n_years:
    print(n_years)
    n_years_label = create_n_years_label(n_years)
    network_to_use = 'empirical_z_scores_%s%s%s'%(regression_label, n_years_label, class_system)
    networks = store[network_to_use]
    networks.fillna(0,inplace=True) #Interpreting nan z-scores as 0
    ### Make sure the networks' arrays are ordered in the same sequence that our classes are ordered
    class_id_sequence = classes_lookup.sort(['Class_ID'])['Class_Name']
    networks = networks.reindex(major_axis=class_id_sequence, 
                                minor_axis=class_id_sequence)
    
    for connectivity_metric in networks.labels:
        print(connectivity_metric)
        for combination_method in ['mean']:#, 'max']:
            connectivity_metric_suffix = create_n_years_label(n_years,suffix=True)
            calculate_connectivities(connectivity_metric,
                                     connectivity_metric_suffix,
                                     networks=networks,
                                     use_the_future=False,
                                     combination_method=combination_method,
                                     additional_label=summary_statistic_label
                                    )
#             rankify(connectivity_metric+connectivity_metric_suffix+'_'+summary_statistic_label+'_'+combination_method, new_class_data)   


# In[39]:

# summary_statistic_label = 'percent_positive'
# regression_label = ''

# for n_years in all_n_years:
#     print(n_years)
#     n_years_label = create_n_years_label(n_years)
#     network_to_use = 'empirical_z_scores_%s%s%s'%(regression_label, n_years_label, class_system)
#     networks = store[network_to_use]
#     networks.fillna(0,inplace=True) #Interpreting nan z-scores as 0
#     ### Make sure the networks' arrays are ordered in the same sequence that our classes are ordered
#     class_id_sequence = classes_lookup.sort(['Class_ID'])['Class_Name']
#     networks = networks.reindex(major_axis=class_id_sequence, 
#                                 minor_axis=class_id_sequence)
#     networks = networks>0
#     networks = networks.cumsum(axis=1)
    
#     n = networks.copy()
#     for i in arange(len(n.items)):
#         n.iloc[:,i] = i+1
#     networks = networks.divide(n)
    
#     for connectivity_metric in networks.labels:
#         print(connectivity_metric)
#         for combination_method in ['mean']:#, 'max']:
#             connectivity_metric_suffix = create_n_years_label(n_years,suffix=True)
#             calculate_connectivities(connectivity_metric,
#                                      connectivity_metric_suffix,
#                                      networks=networks,
#                                      use_the_future=False,
#                                      combination_method=combination_method,
#                                      additional_label=summary_statistic_label
#                                     )
# #             rankify(connectivity_metric+connectivity_metric_suffix+'percent_positive'+combination_method, new_class_data)   


# In[44]:

summary_statistic_label = 'percent_positive'
regression_label = ''

n_years = 1
n_years_label = create_n_years_label(n_years)
network_to_use = 'empirical_z_scores_%s%s%s'%(regression_label, n_years_label, class_system)
networks = store[network_to_use]
networks.fillna(0,inplace=True) #Interpreting nan z-scores as 0
### Make sure the networks' arrays are ordered in the same sequence that our classes are ordered
class_id_sequence = classes_lookup.sort(['Class_ID'])['Class_Name']
networks = networks.reindex(major_axis=class_id_sequence, 
                            minor_axis=class_id_sequence)
networks = networks>0


for n_years_back in ['all']: #all_n_years:
    print(n_years_back)
    if n_years_back==1:
        these_networks = networks
    elif n_years_back=='all':
#         continue #We already calculated this in the cell above.
        these_networks = networks.cumsum(axis=1)
        r = these_networks.copy()        
        n = these_networks.copy()
        for i in arange(len(n.items)):
            n.iloc[:,i] = i+1
        these_networks = r.divide(n)

    else:
        these_networks = networks.cumsum(axis=1)
        
        r = these_networks.copy()
        #Lower the sums to only consider n_years_back
        s = (r.iloc[:,n_years_back:].values-r.iloc[:,:-n_years_back].values) 
        r.iloc[:,n_years_back:] = s
        
        n = these_networks.copy()
        for i in arange(len(n.items)):
            n.iloc[:,i] = min(i+1, n_years_back)
        these_networks = r.divide(n)

    
    for connectivity_metric in networks.labels:
        print(connectivity_metric)
        for combination_method in ['mean']:#, 'max']:
            connectivity_metric_suffix = create_n_years_label(n_years,suffix=True)
            calculate_connectivities(connectivity_metric,
                                     connectivity_metric_suffix,
                                     networks=these_networks,
                                     use_the_future=False,
                                     combination_method=combination_method,
                                     additional_label=summary_statistic_label+'_%s_years_back'%str(n_years_back)
                                    )
    #       rankify(connectivity_metric+connectivity_metric_suffix+'percent_positive'+combination_method, new_class_data)   


# In[41]:

summary_statistic_label = 'percent_positive'
regression_label = ''

n_years = 1
n_years_label = create_n_years_label(n_years)
network_to_use = 'empirical_z_scores_%s%s%s'%(regression_label, n_years_label, class_system)
networks = store[network_to_use]
networks.fillna(0,inplace=True) #Interpreting nan z-scores as 0
### Make sure the networks' arrays are ordered in the same sequence that our classes are ordered
class_id_sequence = classes_lookup.sort(['Class_ID'])['Class_Name']
networks = networks.reindex(major_axis=class_id_sequence, 
                            minor_axis=class_id_sequence)
networks_unthresholded = networks.copy()

for threshold in [1,2,3,5,6,9]:
    networks = networks_unthresholded>threshold

    for n_years_back in ['all']:#all_n_years:
        print(n_years_back)
        if n_years_back==1:
            these_networks = networks
        elif n_years_back=='all':
    #         continue #We already calculated this in the cell above.
            these_networks = networks.cumsum(axis=1)
            r = these_networks.copy()        
            n = these_networks.copy()
            for i in arange(len(n.items)):
                n.iloc[:,i] = i+1
            these_networks = r.divide(n)

        else:
            these_networks = networks.cumsum(axis=1)

            r = these_networks.copy()
            #Lower the sums to only consider n_years_back
            s = (r.iloc[:,n_years_back:].values-r.iloc[:,:-n_years_back].values) 
            r.iloc[:,n_years_back:] = s

            n = these_networks.copy()
            for i in arange(len(n.items)):
                n.iloc[:,i] = min(i+1, n_years_back)
            these_networks = r.divide(n)


        for connectivity_metric in networks.labels:
            print(connectivity_metric)
            for combination_method in ['mean']:#, 'max']:
                connectivity_metric_suffix = create_n_years_label(n_years,suffix=True)
                calculate_connectivities(connectivity_metric,
                                         connectivity_metric_suffix,
                                         networks=these_networks,
                                         use_the_future=False,
                                         combination_method=combination_method,
                                         additional_label=summary_statistic_label+'_%s_years_back_thr_%iz'%(str(n_years_back),
                                                                                                            threshold)
                                                                                                           
                                        )
        #       rankify(connectivity_metric+connectivity_metric_suffix+'percent_positive'+combination_method, new_class_data)   


# In[42]:

summary_statistic_label = 'percent_positive'
regression_label = 'regressed_'

n_years = 1
n_years_label = create_n_years_label(n_years)
network_to_use = 'empirical_z_scores_%s%s%s'%(regression_label, n_years_label, class_system)
networks = store[network_to_use]
networks.fillna(0,inplace=True) #Interpreting nan z-scores as 0
### Make sure the networks' arrays are ordered in the same sequence that our classes are ordered
class_id_sequence = classes_lookup.sort(['Class_ID'])['Class_Name']
networks = networks.reindex(major_axis=class_id_sequence, 
                            minor_axis=class_id_sequence)
networks_unthresholded = networks.copy()

for threshold in [1,2,3,5,6,9]:
    networks = networks_unthresholded>threshold

    for n_years_back in ['all']:#all_n_years:
        print(n_years_back)
        if n_years_back==1:
            these_networks = networks
        elif n_years_back=='all':
    #         continue #We already calculated this in the cell above.
            these_networks = networks.cumsum(axis=1)
            r = these_networks.copy()        
            n = these_networks.copy()
            for i in arange(len(n.items)):
                n.iloc[:,i] = i+1
            these_networks = r.divide(n)

        else:
            these_networks = networks.cumsum(axis=1)

            r = these_networks.copy()
            #Lower the sums to only consider n_years_back
            s = (r.iloc[:,n_years_back:].values-r.iloc[:,:-n_years_back].values) 
            r.iloc[:,n_years_back:] = s

            n = these_networks.copy()
            for i in arange(len(n.items)):
                n.iloc[:,i] = min(i+1, n_years_back)
            these_networks = r.divide(n)


        for connectivity_metric in networks.labels:
            print(connectivity_metric)
            for combination_method in ['mean']:#, 'max']:
                connectivity_metric_suffix = create_n_years_label(n_years,suffix=True)
                calculate_connectivities(connectivity_metric,
                                         connectivity_metric_suffix,
                                         networks=these_networks,
                                         use_the_future=False,
                                         combination_method=combination_method,
                                         additional_label=summary_statistic_label+'_%s_years_back_thr_%iz_regressed'%(str(n_years_back),
                                                                                                            threshold)
                                                                                                           
                                        )
        #       rankify(connectivity_metric+connectivity_metric_suffix+'percent_positive'+combination_method, new_class_data)   


# In[43]:

store.close()


# Make citation count predictors
# ====

# Class citation counts

# In[44]:

if use_precalculated_supporting_data:
    class_citation_count = pd.read_hdf(data_directory+'class_citation_counts.h5', 
                                              class_system)

    class_citation_count_lookup = pd.read_hdf(data_directory+'class_citation_counts.h5', 
                                              'class_lookup_table_%s'%class_system)

    class_citation_count['Class_ID'] = classes_lookup.set_index('Class_Name').ix[         class_citation_count_lookup.ix[class_citation_count['Class_ID'], 'index'].values                                                                                 ]['Class_ID'].values


    measures = [m for m in class_citation_count if m not in ['Class_ID', 'Issued_Year']]
    class_citation_count.rename(columns=dict([(m, m+'_Previous_Year') for m in measures]), inplace=True)
    
    class_citation_count.rename(columns={'Issued_Year': 'Application_Year'}, inplace=True)
    class_citation_count['Application_Year'] = class_citation_count['Application_Year']+1
    #We're calling this "Application_Year", but really it's the application year it will be paired with.
    #It's the application year that will have this information present. 
    
    new_class_data = new_class_data.merge(class_citation_count, on=['Class_ID', 'Application_Year'], how='left').fillna(0)
    for m in measures: 
        rankify(m+'_Previous_Year',data=new_class_data)
        
else:
    store = pd.HDFStore(data_directory+'citations_organized.h5')
    citations = store['citations']
    citation_class_lookup = store['%s_class_lookup'%class_system]
    citation_class_lookup = citation_class_lookup.reset_index().set_index(0)
    store.close()

    for column in citations.columns:
        if class_system in column:
            new_name = column.replace('_'+class_system, "")
            citations.rename(columns={column: new_name}, inplace=True)

    class_citations_dict = {}

    for citation_type, class_column in [('Outward', 'Class_Citing_Patent'),
                                        ('Inward', 'Class_Cited_Patent')]:
        count_by_year = citations.groupby([class_column, 'Year_Citing_Patent'])['Citing_Patent'].count().sort_index([class_column, 'Year_Citing_Patent'])
        cumulative_count_by_year = count_by_year.groupby(level=class_column).cumsum().reset_index()
        count_by_year = count_by_year.reset_index()

        class_citations_dict['Class_%s_Citation_Count'%citation_type] = count_by_year.reset_index().rename(columns={class_column:'Class_ID',
                                                                                                                                    'Citing_Patent': 'Count'})
        class_citations_dict['Class_Cumulative_%s_Citation_Count'%citation_type] = cumulative_count_by_year.rename(columns={0:'Count',
                                                                                                                                             class_column:'Class_ID'})
    for k in class_citations_dict.keys():
        class_citations_dict[k].rename(columns={'Year_Citing_Patent': 'Issued_Year'}, inplace=True)

        #The stored citations data may have a different class_lookup index than we have calculated here
        #so we convert it to ours.
        class_citations_dict[k]['Class_ID'] = classes_lookup.set_index('Class_Name').ix[citation_class_lookup.ix[
                class_citations_dict[k]['Class_ID']]['index']]['Class_ID'].values
    for k in class_citations_dict.keys():

        class_citations_dict[k]['Application_Year'] = class_citations_dict[k]['Issued_Year']+1 
        #We're calling this "Application_Year", but really it's the application year it will be paired with.
        #It's the application year that will have this information present. 
        new_class_data[k+'_Previous_Year'] = new_class_data[['Class_ID','Application_Year']].merge(class_citations_dict[k],
                                          on=['Class_ID', 'Application_Year'],
                                          how='left')['Count'].fillna(0)
        rankify(k+'_Previous_Year', new_class_data)


# Agent citation counts to individual classes

# In[45]:

### Agent citation counts to individual classes
def calculate_agent_citation_counts(citation_data,
                                    inventorships,
                                    agents_lookup=agents_lookup,
                                    new_class_data=new_class_data,
                                    new_class_entries=new_class_entries,
                                    ):

    n_rows_to_be_created = new_class_data.shape[0]
    input_data = new_class_entries.sort(['Agent_ID', 
                                         'Agent_Class_Number'])[['Agent_ID', 
                                                                 'Agent_Class_Number', 
                                                                 'Class_ID', 
                                                                 'Application_Year']].values 
    output_data = zeros((n_rows_to_be_created, 4))
    output_line = 0
    for input_line in arange(len(input_data)):
        output_line = multiindex_maker(input_line, input_data, 
                                       output_line, output_data,
                                       function_to_apply=calculate_agent_citation_counts_helper,
                                       function_data=(citation_data,
                                                      agents_lookup,
                                                      inventorships)
                        )

    new_class_data.sort(['Agent_ID', 'Agent_Class_Number'], inplace=True)
    new_class_data['Agent_Previous_Citations_to_Class'] = output_data[:,3]
#     new_class_data['Agent_Previous_Citations_from_Class'] = output_data[:,4]
    return

def calculate_agent_citation_counts_helper(input_line, input_data, 
                                    output_line, output_data,
                                    classes_previously_entered,
                                    classes_available,
                                    n_reps,
                                    other_data):
    
    year = input_data[input_line,3]
    agent = input_data[input_line,0]
    
    patent_class_citation_count, agents_lookup, inventorships = other_data    
    
#     agent = agents_lookup.ix[agent,'Disambiguation_ID']
    agents_patents = inventorships.ix[agent, ['Patent', 'Issued_Year']]
    earlier_agents_patents = agents_patents[agents_patents['Issued_Year']<year]['Patent'].values

    #citations['Citing_Patent'].isin(earlier_agents_patents)
#     earlier_citations_received = citations['Cited_Patent'].isin(earlier_agents_patents)
#     earlier_citations_received = (earlier_citations_received) & (citations['Year_Citing_Patent']<year)
    
#     print(earlier_citations_made)
#     print(earlier_citations_received)
#     agent_citations_to_class = citations[earlier_citations_made]['Class_Cited_Patent'].value_counts()
#     agent_citations_from_class = citations[earlier_citations_received]['Class_Citing_Patent'].value_counts()

#     temp = zeros(n_classes)
#     temp[agent_citations_to_class.index.values] = agent_citations_to_class.values
#     agent_citations_to_class = temp[classes_available]
    
#     temp = zeros(n_classes)
#     temp[agent_citations_from_class.index.values] = agent_citations_from_class.values
#     agent_citations_from_class = temp[classes_available]

    earlier_citations_made = patent_class_citation_count.ix[earlier_agents_patents].fillna(0).sum()

    output_data[output_line:output_line+n_reps,3] = earlier_citations_made[classes_available].values
#     output_data[output_line:output_line+n_reps,3] = agent_citations_from_class
    return    


# In[46]:

if use_precalculated_supporting_data:
    patent_class_citation_count = pd.read_hdf(data_directory+'patent_class_citation_count.h5', 
                                              class_system)
    patent_class_citation_count_class_lookup = pd.read_hdf(data_directory+'patent_class_citation_count.h5',
                                                           'class_lookup_table_%s'%class_system)
    patent_class_citation_count_class_lookup = pd.DataFrame(patent_class_citation_count_class_lookup).reset_index()
else:
    patent_class_citation_count = citations.groupby('Citing_Patent')['Class_Cited_Patent'].value_counts()
    patent_class_citation_count_class_lookup = citation_class_lookup

if store_calculated_supporting_data:
    patent_class_citation_count.to_hdf(data_directory+'patent_class_citation_count.h5', class_system)


# In[47]:

patent_class_citation_count = patent_class_citation_count.unstack()


# In[ ]:

### Convert patent_class_citation_count's class IDs to the same ones we're using here.
patent_class_citation_count.columns = classes_lookup.set_index('Class_Name').ix[patent_class_citation_count_class_lookup.ix[patent_class_citation_count.columns, 
                                                                                                         'index'].values]['Class_ID'].values


# In[ ]:

data.set_index('Agent_ID', inplace=True)
data.sort_index(inplace=True)


# In[ ]:

calculate_agent_citation_counts(patent_class_citation_count, data)


# In[ ]:

# rankify('Agent_Previous_Citations_to_Class')
# rankify('Agent_Previous_Citations_from_Class')


# Make social network predictors
# ====
# "Who of my previous co-authors had patented in that classes, before I patented with them?"
# The most computationally intensive part is identifying all of an author's previous co-authors, and storing them. Once that's done you "just" need to iterate through each row of new_class_entries, identify the author and year for that entry, then lookup the author's co-authors wherever you stored them, then find THEIR previous entries in new_class_entries (entries that were BEFORE you patented with them)

# In[ ]:

if use_precalculated_supporting_data:
    store = pd.HDFStore(data_directory+'agent_patent_relationships.h5',mode='r')
    agent_patent_lists = store['agent_patent_lists']
    agent_patent_year_lists = store['agent_patent_year_lists']
    patent_agent_lists = store['patent_agent_lists']
    patent_classes = store['patent_classes_%s'%class_system]
    patent_classes_lookup = store['class_lookup_table_%s'%class_system]
    store.close()
    
    patent_classes['Class_ID'] = classes_lookup.set_index('Class_Name').ix[         patent_classes_lookup.ix[patent_classes['Class_ID'], 'Class_Name'].values                                                                           ]['Class_ID'].values
    
else:
    agent_patent_lists = all_data.groupby(level='Agent')['Patent'].apply(lambda x: list(x))
    agent_patent_year_lists = all_data.groupby(level='Agent')['Issued_Year'].apply(lambda x: list(x))
    patent_agent_lists = all_data.reset_index().groupby('Patent')['Agent'].apply(lambda x: list(x))
    patent_classes = all_data.drop_duplicates('Patent')[['Patent', 'Application_Year', 'Class_ID']].set_index('Patent')

if store_calculated_supporting_data:
    store = pd.HDFStore(data_directory+'agent_patent_relationships.h5',mode='w')
    store.put('/agent_patent_lists', agent_patent_lists)
    store.put('/agent_patent_year_lists', agent_patent_year_lists)
    store.put('/patent_agent_lists', patent_agent_lists)
    store.put('/patent_classes_%s'%class_system, patent_classes)
    store.close()


# In[ ]:

### Previous co-agent patenting counts in individual classes 
def calculate_co_agent_class_counts(agent_patent_lists,
                                    agent_patent_year_lists,
                                    patent_agent_lists,
                                    patent_classes,
                                    agents_lookup=agents_lookup,
                                    new_class_data=new_class_data,
                                    new_class_entries=new_class_entries,
                                    ):

    n_rows_to_be_created = new_class_data.shape[0]
    input_data = new_class_entries.sort(['Agent_ID', 
                                         'Agent_Class_Number'])[['Agent_ID', 
                                                                 'Agent_Class_Number', 
                                                                 'Class_ID', 
                                                                 'Application_Year']].values 
    output_data = zeros((n_rows_to_be_created, 5))
    output_line = 0
    for input_line in arange(len(input_data)):
        output_line = multiindex_maker(input_line, input_data, 
                                       output_line, output_data,
                                       function_to_apply=calculate_co_agent_class_counts_helper,
                                       function_data=(agents_lookup, 
                                                      agent_patent_lists, 
                                                      agent_patent_year_lists, 
                                                      patent_agent_lists, 
                                                      patent_classes)
                        )

    new_class_data.sort(['Agent_ID', 'Agent_Class_Number'], inplace=True)
    new_class_data['CoAgent_Previous_Patent_Count_in_Class'] = output_data[:,3]
    new_class_data['CoAgent_Count_in_Class'] = output_data[:,4]

    return

def calculate_co_agent_class_counts_helper(input_line, input_data, 
                                    output_line, output_data,
                                    classes_previously_entered,
                                    classes_available,
                                    n_reps,
                                    other_data):
    
    year = input_data[input_line,3]
    agent = input_data[input_line,0]
    
    agents_lookup, agent_patent_lists, agent_patent_year_lists, patent_agent_lists, patent_classes = other_data    
    
    agent = agents_lookup.ix[agent,'Disambiguation_ID']
 
    patents_of_agent = agent_patent_lists.ix[agent]
    years_of_patents_of_agent = array(agent_patent_year_lists.ix[agent])
    earlier_patent_indices = years_of_patents_of_agent<year
    earlier_patents_of_agent = array(patents_of_agent)[earlier_patent_indices]
    years_of_earlier_patents_of_agent = years_of_patents_of_agent[earlier_patent_indices]

    #Find the sorting to get the most recent patents first
    most_recent_sort = argsort(years_of_earlier_patents_of_agent)[::-1] 

    #Get the agents associated with each patent, sorted with most recent patent first. Note each patent returns a 
    #list of co-agents, and a co-agent may appear again later in the sequence in an earlier patent.
    co_agents_of_earlier_patents = patent_agent_lists.ix[earlier_patents_of_agent[most_recent_sort]]


    previous_co_agents = [agent]
    previous_patents_of_co_agents = []
    previous_classes_of_co_agents = []

    for co_agents_with_this_patent, this_patent_year in zip(co_agents_of_earlier_patents, 
                                         years_of_earlier_patents_of_agent[most_recent_sort]):
        for this_co_agent in co_agents_with_this_patent:
            if this_co_agent not in previous_co_agents:
                previous_co_agents.append(this_co_agent)
                #Because we sorted the years from high to low, we record the most recent instance of an agent first. Therefore,
                #if we already have the agent in the list, that is the best value, and we don't need to record anything more.

                patents_of_co_agent = agent_patent_lists.ix[this_co_agent]
                years_of_patents_of_co_agent = agent_patent_year_lists.ix[this_co_agent]
                earlier_patents_of_co_agent = [patents_of_co_agent[i] for i in range(len(patents_of_co_agent))
                                               if years_of_patents_of_co_agent[i]<this_patent_year]
                previous_patents_of_co_agents += earlier_patents_of_co_agent
                previous_classes_of_co_agents += list(patent_classes.ix[earlier_patents_of_co_agent, 
                                                                        'Class_ID'].dropna().unique())

    class_co_agent_patent_counts = patent_classes.ix[unique(previous_patents_of_co_agents), 
                                                     'Class_ID'].dropna().value_counts()
    temp = zeros(n_classes)
    temp[class_co_agent_patent_counts.index.values.astype('int64')] = class_co_agent_patent_counts.values
    class_co_agent_patent_counts = temp[classes_available]

    class_co_agent_counts = pd.value_counts(previous_classes_of_co_agents)
    temp = zeros(n_classes)
    temp[class_co_agent_counts.index.values.astype('int64')] = class_co_agent_counts.values
    class_co_agent_counts = temp[classes_available]


    output_data[output_line:output_line+n_reps,3] = class_co_agent_patent_counts
    output_data[output_line:output_line+n_reps,4] = class_co_agent_counts
    
    return    


# In[ ]:

### This currently measures both the number of inventions in each class that are near to the agent. 
### (I.e. the number of patents in each class that the agent's co-agents worked on.)
### AND the number of co-agents that had previously been active in each class. 
### This could be smaller than the number of patents or larger (if you've many collaborators who all worked on one patent
### together vs. if you have one collaborator who had many patents before)
### Calculating this second quantity roughly doubles the calculation time. Could possibly be optimized.
calculate_co_agent_class_counts(agent_patent_lists,
                                agent_patent_year_lists,
                                patent_agent_lists,
                                patent_classes)


# In[ ]:

# rankify('CoAgent_Previous_Patent_Count_in_Class')
# rankify('CoAgent_Count_in_Class')


# In[ ]:

new_class_data['Entered'] = new_class_data['Entered'].astype('uint8')


# In[ ]:

new_class_data.drop(['Application_Date'], axis=1, inplace=True)


# In[ ]:

for c in new_class_data.columns:
    if new_class_data[c].dtype == 'float64':
        new_class_data[c] = new_class_data[c].astype('float32')
gc.collect()


# In[ ]:

filename = 'agent_entry_data_%s_'%class_system

if type(agent_sample)==int:
    filename+=('sample_%i_agents'%agent_sample)
elif agent_sample is not None:
    filename+=('sample_%i_%i_agents'%agent_sample)
else:
    filename+='_other'
    
if not use_regressed_z_scores:
    filename+='_unregressed_z_scores'
    
filename+='.h5'

store = pd.HDFStore(data_directory+'Agent_Entries/samples/'+filename, complib='blosc',complevel=9, append=False)
store['all_available_classes'] = new_class_data
store['entries'] = new_class_data[new_class_data.Entered==1]
store['class_lookup'] = classes_lookup
store.close()

