import glob
import numpy as np
import pandas as pd
import pickle
import datetime
import os
import sys
sys.path.append(
    os.path.join("..", "Libraries", "QML_lib")
)
import GrowthRules

results_directory = str(
    os.getcwd() + '/Results/'
)


now = datetime.datetime.now()
hour = now.strftime("%H")
minute = now.strftime("%M")
month = now.strftime("%b")
day = now.strftime("%d")

output_directory = "SharedResults/{}_{}/{}_{}".format(month, day, hour, minute)


# All results

test_run = False
use_all_directories = False
all_directories = {
    'sim_1': 'Oct_07/17_09',
    'sim_2': 'Oct_07/17_11',
    'experimental_data_plusphase': 'Oct_02/18_01',
    'experimental_data_plusrandom': 'Oct_02/18_16',
    'simulation_plusphase': 'Oct_03/17_43',
    'simulation_extended_true_model_plusphase': 'Oct_02/18_18',
    'vary_model_3_params': 'Oct_04/18_15',
    'vary_model_4_params': 'Oct_04/18_17',
    'vary_model_5_params': 'Oct_04/18_18',
    'vary_model_6_params': 'Oct_07/15_24',
    'vary_model_7_params': 'Oct_07/15_27'
}
if use_all_directories == True:
    directories_to_use = list(all_directories.keys())
elif test_run == True:
    directories_to_use = [
        'sim_1', 'sim_2'  # for testing
    ]
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
directories = [str(results_directory + d + '/') for d in directories]

all_results_paths = [glob.glob(d + 'results*') for d in directories]


def flatten(l): return [item for sublist in l for item in sublist]


all_results_paths = flatten(all_results_paths)

results_df = pd.DataFrame()
directory_df = pd.DataFrame()
expectation_values_df = pd.DataFrame()
volume_df = pd.DataFrame()

for d in directories:
    print("Analysing directory ", d)
    idx = directories.index(d)
    true_params_path = str(d + 'true_params.p')
    true_params = pickle.load(open(true_params_path, 'rb'))

    true_expec_vals = pickle.load(
        open(
            str(d + 'true_expec_vals.p'),
            'rb'
        )
    )
    exp_val_times = list(sorted(true_expec_vals.keys()))
    raw_expec_vals = [
        true_expec_vals[t] for t in exp_val_times
    ]

    growth_gen = true_params['growth_generator']
    growth_class = GrowthRules.get_growth_generator_class(
        growth_generation_rule=growth_gen
    )

    directory_info = {
        'Run': directories_to_use[idx],
        'RunIdx': idx,
        'ResultsDirectory': d,
        'TrueModel': true_params['true_op'],
        'TrueParams': true_params['params_dict'],
        'GrowthGenerator': growth_gen,
        'TrueModelLatex': growth_class.latex_name(
            true_params['true_op']
        ),
        'TrueExpectationValues': raw_expec_vals,
        'ExpValTimes': exp_val_times
    }

    d += '/'
    directory_data = pd.Series(directory_info)
    directory_df = directory_df.append(
        directory_data,
        ignore_index=True
    )

    results_paths = glob.glob(d + 'results*')

    for results_file in results_paths:
        res = pickle.load(open(results_file, 'rb'))
        data = pd.Series(res)
        data['ResultsDirectory'] = d
        data['RunIdx'] = idx
        data['ExpValTimes'] = sorted(res['ExpectationValues'].keys())
        data['RawExpVals'] = [
            res['ExpectationValues'][t] for t in data['ExpValTimes']
        ]
        epochs = sorted(res['TrackVolume'].keys())
        data['RawVolumes'] = [res['TrackVolume'][e] for e in epochs]
        data['Epochs'] = epochs
        results_df = results_df.append(
            data,
            ignore_index=True
        )

results_file = "{}/combined_results.csv".format(output_directory)
run_info_file = "{}/run_info.csv".format(output_directory)


if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print("Storing results dataframe in {}".format(results_file))
results_df.to_csv(
    results_file
)
print("Storing run info dataframe in {}".format(run_info_file))

directory_df.to_csv(
    run_info_file
)


# True expectation Values
true_expectation_values = pd.DataFrame()
run_identifiers = [int(r) for r in directory_df.index.values]
for run_id in run_identifiers:
    true_exp_vals = directory_df.iloc[run_id].TrueExpectationValues
    true_times = directory_df.iloc[run_id].ExpValTimes
    for t, e in zip(true_times, true_exp_vals):
        true_exp_vals = pd.Series(
            {
                'RunIdx': run_id,
                't': t,
                'TrueExpVal': e
            }
        )

        true_expectation_values = true_expectation_values.append(
            true_exp_vals,
            ignore_index=True
        )

true_expectation_values_path = "{}/true_expectation_values.csv".format(
    output_directory)
true_expectation_values.to_csv(
    true_expectation_values_path
)


# Expectation Values

# ex_val_df = pd.DataFrame()
# for k in list(results_df.index):
#     row = results_df.iloc[k]
#     a = [
#             pd.Series(
#             {
#                 'RunIdx' : row.RunIdx,
#                 'ChampLatex' : row.champion_name_latex,
#                 'time' : t,
#                 'ExpVal' : e
#             }
#         )
#         for t,e in zip(row.ExpValTimes, row.RawExpVals)
#     ]
#     ex_val_df = ex_val_df.append(
#         pd.DataFrame(a),
#         ignore_index=True
#     )
# expectation_values_path = "{}/expectation_values.csv".format(results_directory)
# print("Storing expectation values df in {}".format(expectation_values_path))
# ex_val_df.to_csv(
#     expectation_values_path
# )

# # Volumes
# vol_df = pd.DataFrame()
# for k in list(results_df):
#     row = results_df.iloc[k]
#     a = [
#             pd.Series(
#             {
#                 'RunIdx' : row.RunIdx,
#                 'ChampLatex' : row.champion_name_latex,
#                 'Epoch' : e,
#                 'Volume' : v
#             }
#         )
#         for e,v in zip(row.Epochs, row.RawVolumes)
#     ]
#     print(a)
#     vol_df = vol_df.append(
#         pd.DataFrame(a),
#         ignore_index=True
#     )

# volumes_path = "{}/volumes.csv".format(output_directory)
# vol_df.to_csv(
#     volumes_path
# )
