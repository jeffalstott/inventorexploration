
# coding: utf-8

# In[1]:

# class_system = 'IPC4'
# data_directory = '../data/'
# sample_length = 1000
# samples = range(0,773000,sample_length)
# rescale_z = True

# from os import path
# import sys
# python_location = path.dirname(sys.executable)+'/python'

# from os import path
# abs_path_data_directory = path.abspath(data_directory)+'/'


# In[ ]:

# relatedness_types = [ 'Class_Cites_Class_Count_1_years_percent_positive_all_years_back_mean',
#                      'Class_Cited_by_Class_Count_1_years_percent_positive_all_years_back_mean',
#                      'Class_CoOccurrence_Count_Inventor_1_years_percent_positive_all_years_back_mean',
#                      'Class_CoOccurrence_Count_PID_1_years_percent_positive_all_years_back_mean',
#                     ]


# In[4]:

basic_program = open('Fit_PDF_4D_to_Samples.py', 'r').read()


# In[6]:

from os import system

for sample_start in samples:
    header = """#!{2}
#PBS -l nodes=1:ppn=1
#PBS -l walltime=0:20:00
#PBS -l mem=10000m
#PBS -N sample_{0}_{1}
""".format(sample_start, class_system, python_location)
    
    options = """sample_start={0}
sample_length={1}
class_system = '{2}'
data_directory = '{3}'
rescale_z = {4}
relatedness_types = {5}
""".format(sample_start, sample_length,
          class_system,
          abs_path_data_directory,
          rescale_z,
           relatedness_types
          )

    this_program = header+options+basic_program
    this_job_file = 'jobfiles/Fit_PDF_sample_{0}_{1}_{2}.py'.format(sample_start, 
                                                            sample_start+sample_length, 
                                                            class_system)


    f = open(this_job_file, 'w')
    f.write(this_program)
    f.close()

    system('qsub '+this_job_file)

