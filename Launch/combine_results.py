import glob
import numpy as np
import pandas as pd
import pickle
import datetime
import os

results_directory = str(
	os.getcwd() + '/Results/'
)
use_all_directories = False
all_directories = {
	'sim_1' : 'Oct_07/17_09',
	'sim_2' : 'Oct_07/17_11',
	'experimental_data_plusphase' : 'Oct_02/18_01',
	'experimental_data_plusrandom' : 'Oct_02/18_16',
	'simulation_plusphase' : 'Oct_03/17_43',
	'simulation_extended_true_model_plusphase' : 'Oct_02/18_18',
	'vary_model_3_params' : 'Oct_04/18_15',
	'vary_model_4_params' : 'Oct_04/18_17',
	'vary_model_5_params' : 'Oct_04/18_18',
	'vary_model_6_params' : 'Oct_07/15_24',
	'vary_model_7_params' : 'Oct_07/15_27'
}

if use_all_directories == True:
	directories_to_use = list(all_directories.keys())
else:
	directories_to_use = [
		# 'sim_1', 'sim_2' # for testing
		'experimental_data_plusphase',
		'experimental_data_plusrandom',
		'simulation_plusphase',
		'simulation_extended_true_model_plusphase',
		'vary_model_3_params',
		'vary_model_4_params',
		'vary_model_5_params',
		'vary_model_6_params',
		'vary_model_7_params',
	]

directories = [all_directories[d] for d in directories_to_use]
directories = [ str(results_directory + d + '/') for d in directories]

all_results_paths = [glob.glob(d+'results*') for d in directories]

flatten = lambda l: [item for sublist in l for item in sublist]
all_results_paths = flatten(all_results_paths)

results_df = pd.DataFrame()

for d in directories:
	idx = directories.index(d)
	d += '/'
	description = directories_to_use[idx]
	results_paths = glob.glob(d+'results*')
	true_params_path = str(d + 'true_params.p')
	true_params = pickle.load(open(true_params_path, 'rb'))
	true_op = true_params['true_op']
	growth_gen = true_params['growth_generator']

	for results_file in results_paths:
	    res = pickle.load(open(results_file, 'rb'))
	    data = pd.Series(res)
	    data['TrueModel'] = true_op
	    data['GrowthGenerator'] = growth_gen
	    data['ResultsDirectory'] = d
	    data['Run'] = description
	    results_df = results_df.append(
	        data, 
	        ignore_index=True
	    )


# for results_file in all_results_paths:
#     res = pickle.load(open(results_file, 'rb'))
#     data = pd.Series(res)
#     data['ResultsDirectory'] = results_file[:-14]
#     results_df = results_df.append(
#         data, 
#         ignore_index=True
#     )



now = datetime.datetime.now()
hour = now.strftime("%H")
minute = now.strftime("%M")
month = now.strftime("%b")
day = now.strftime("%d")

results_directory = "SharedResults/{}_{}/{}_{}".format(month, day, hour, minute)
results_file = "{}/combined_results.csv".format(results_directory)

if not os.path.exists(results_directory):
	os.makedirs(results_directory)

print("Storing results dataframe in {}".format(results_file))
results_df.to_csv(
	results_file
)

