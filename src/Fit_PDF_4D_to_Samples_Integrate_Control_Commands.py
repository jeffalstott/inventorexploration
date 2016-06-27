
# coding: utf-8

# In[29]:

from os import system


# In[30]:

basic_program = open('Fit_PDF_4D_to_Samples_Integrate.py', 'r').read()


# In[31]:

# max_sample = 773000
# sample_length = 1000
# sample_start = 0
# n_combine = 100000
# class_system = 'IPC4'
# rescale_z = True

# from os import path
# import sys
# python_location = path.dirname(sys.executable)+'/python'
# data_directory = '../data/'
# abs_path_data_directory = path.abspath(data_directory)+'/'


# In[32]:

# relatedness_types = [ 'Class_Cites_Class_Count_1_years_percent_positive_all_years_back_mean',
#                      'Class_Cited_by_Class_Count_1_years_percent_positive_all_years_back_mean',
#                      'Class_CoOccurrence_Count_Inventor_1_years_percent_positive_all_years_back_mean',
#                      'Class_CoOccurrence_Count_PID_1_years_percent_positive_all_years_back_mean',
#                     ]


# In[35]:

popularity_types = ['Class_Patent_Count_1_years_Previous_Year_Percentile']


# In[36]:

for relatedness in relatedness_types:
    for popularity in popularity_types:
        header = """#!{2}
#PBS -l nodes=1:ppn=5
#PBS -l walltime=0:45:00
#PBS -l mem=20000m
#PBS -N PDF_{0}_{1}
""".format(relatedness, popularity, python_location)
    
        options = """relatedness_types = ['{0}']
popularity_types = ['{1}']
data_directory = '{2}'
class_system = '{3}'
max_sample = {4}
sample_start = {5}
sample_length = {6}
n_combine = {7}
rescale_z = {8}
""".format(relatedness, popularity,
           abs_path_data_directory,
          class_system,
           max_sample,
           sample_start,
           sample_length,
           n_combine,
           rescale_z
          )

    this_program = header+options+basic_program
    this_job_file = 'jobfiles/PDF_{0}_{1}_{2}.py'.format(relatedness, popularity, class_system)


    f = open(this_job_file, 'w')
    f.write(this_program)
    f.close()

    system('qsub '+this_job_file)

