
# coding: utf-8

# In[1]:

# import pandas as pd
# %pylab inline


# In[101]:

# data_directory = '../data/'
# class_systems = ['IPC', 'IPC4']
# all_n_years = ['all', 1, 5]

# def create_n_years_label(n_years):
#     if n_years is None or n_years=='all' or n_years=='cumulative':
#         n_years_label = ''
#     else:
#         n_years_label = '%i_years_'%n_years
#     return n_years_label


# In[93]:

def normalize_out(target, norm_out):
    x, y = norm_out.values.ravel(), target.values.ravel()
    x = x[y>0]
    y = y[y>0]
    y_hat = regress(x,y)

    norm_y_up = y/y_hat

    ####
    x, y = norm_out.values.ravel(), target.values.ravel()
    x = x[y<0]
    y = abs(y[y<0])
    y_hat = regress(x,y)
    norm_y_down = y/y_hat

    target_norm = target.copy()
    target_norm.values[target_norm.values>0] = norm_y_up
    target_norm.values[target_norm.values<0] = -norm_y_down
    return target_norm

from scipy.stats import linregress

def regress(x,y):
    slope, intercept, r, p, stderr = linregress(log(x),log(y))
    norm = lambda x: x**slope * exp(intercept)
    return norm(x)


# In[97]:

def normalize_out(target, norm_out):
    x, y = norm_out.values.ravel(), target.values.ravel()
    valid_ind_up = (y>0) & (x>0)
    x = x[valid_ind_up]
    y = y[valid_ind_up]
    y_hat = regress(x,y)

    norm_y_up = y/y_hat

    ####
    x, y = norm_out.values.ravel(), target.values.ravel()
    valid_ind_down = (y<0) & (x>0)
    x = x[valid_ind_down]
    y = abs(y[valid_ind_down])
    y_hat = regress(x,y)
    norm_y_down = y/y_hat

    target_norm = target.copy()
    target_norm.values[(target_norm.values>0) & (norm_out.values>0)] = norm_y_up
    target_norm.values[(target_norm.values<0) & (norm_out.values>0)] = -norm_y_down
    return target_norm

from scipy.stats import linregress

def regress(x,y):
    slope, intercept, r, p, stderr = linregress(log(x),log(y))
    norm = lambda x: x**slope * exp(intercept)
    return norm(x)


# In[91]:

popularities = pd.HDFStore(data_directory+'popularity_networks.h5')
relatedness = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5')


# In[102]:

for class_system in class_systems:
    print(class_system)
    for n_years in all_n_years:
        print(n_years)
        n_years_label = create_n_years_label(n_years)
        r = relatedness['empirical_z_scores_%s%s'%(n_years_label, class_system)].fillna(0)
        p = popularities['patent_count_%s%s'%(n_years_label, class_system)].ix[:,r.major_axis, r.major_axis]
        r_regressed = r.copy()
        for label in r.labels:
            for item in r.items:
                r_regressed.ix[label,item] = normalize_out(r.ix[label,item], 
                                                             p.ix[item])
        print(any(r_regressed.isnull()))
        relatedness.put('/empirical_z_scores_regressed_%s%s'%(n_years_label,
                                                class_system), r_regressed, 'table', append=False)
        
        


# In[103]:

relatedness.close()
popularities.close()

