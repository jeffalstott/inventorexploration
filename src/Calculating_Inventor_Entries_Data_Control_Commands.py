
# coding: utf-8

# In[ ]:

# job_files_directory = 'jobfiles/'
# sample_length = 10000
# all_samples_end = 690000
# all_samples_start = 0
# abs_path_data_directory = ...


# In[ ]:

if overwrite:
    samples = range(all_samples_start,all_samples_end,sample_length)
else:
    from os import listdir
    dirlist = listdir(data_directory+'Agent_Entries/samples/')
    from pylab import *
    unrun_samples = ones(len(range(all_samples_start,all_samples_end,sample_length)))

    for f in dirlist:
        if f.startswith('agent_entry_data_%s_sample_'%class_system):
            n = int(f.split('_')[-3])/sample_length
            unrun_samples[n] = 0

    unrun_samples = where(unrun_samples)[0]*sample_length

    samples = unrun_samples


# In[ ]:

from os import system
basic_program = open('Calculating_Inventor_Entries_Data.py', 'r').read()


# In[ ]:

for sample_start in samples:
    header = """#!{2}
#PBS -l nodes=1:ppn=5
#PBS -l walltime=72:00:00
#PBS -l mem=40000m
#PBS -N sample_{0}_{1}
""".format(sample_start, class_system, python_location)
    
    options = """agent_sample=({0}, {1})
print("Sample range: %i to %i"%agent_sample)
class_system = '{2}'
all_n_years = {3}
use_precalculated_supporting_data = True
store_calculated_supporting_data = False
use_regressed_z_scores = True
data_directory = '{4}'
""".format(sample_start, sample_start+sample_length,
          class_system,
          all_n_years,
          abs_path_data_directory)

    this_program = header+options+basic_program
    this_job_file = 'jobfiles/sample_{0}_{1}_{2}.py'.format(sample_start, 
                                                            sample_start+sample_length, 
                                                            class_system)


    f = open(this_job_file, 'w')
    f.write(this_program)
    f.close()

    system('qsub '+this_job_file)

