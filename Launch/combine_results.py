import glob
import numpy as np
import pandas as pd
import pickle
import datetime
import os

# results_directory = '/panfs/panasas01/phys/bf16951/QMD/Launch/Results/'
# results_directory = '/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations/Results/'
results_directory = str(
	os.getcwd() + '/Results/'
)
all_directories = {
	'sim_1' : 'Oct_07/17_09',
	'sim_2' : 'Oct_07/17_11',
	'vary_model_3_params' : 'Oct_04/18_15',
	'vary_model_4_params' : 'Oct_04/18_17',
	'vary_model_5_params' : 'Oct_04/18_18',
	'vary_model_6_params' : 'Oct_07/15_24',
	'vary_model_7_params' : 'Oct_07/15_27'
}

directories_to_use = [
	# 'sim_1', 'sim_2' # for testing
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
for results_file in all_results_paths:
    res = pickle.load(open(results_file, 'rb'))
    data = pd.Series(res)
    data['ResultsDirectory'] = results_file[:-14]
    results_df = results_df.append(
        data, 
        ignore_index=True
    )



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

