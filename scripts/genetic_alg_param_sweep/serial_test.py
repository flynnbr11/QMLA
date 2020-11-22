import sys
import os
import numpy as np
import pandas as pd
import time
import pickle
import datetime

from param_sweep import get_all_configurations, run_genetic_algorithm


ga_results_df = pd.DataFrame() 

# get configurations to cycle over
all_configurations = get_all_configurations(
    log_file = os.path.join(
        os.getcwd(), 'output.log'
    ),
)
iterable_configurations = iter(all_configurations)
print("[MASTER] Configurations to cycle through:", all_configurations, flush=True)

while True:
    try:
        configuration = next(iterable_configurations)
    except:
        print("No configurations left.")
        break

    print("Starting configuration:", configuration)
    result = run_genetic_algorithm(configuration = configuration)
    print("Finished, result:", result)
    ga_results_df = ga_results_df.append(
        result, 
        ignore_index=True
    )



# store the result
result_directory = os.path.join(os.getcwd(), 'results')
if not result_directory: os.makedirs(result_directory)

now = datetime.datetime.now()
time = "{}_{}_{}_{}".format(
    now.strftime("%b"),
    now.strftime("%d"),
    now.strftime("%H"),
    now.strftime("%M"),
)
path_to_store_result = os.path.join(
    result_directory, 
    'results_{}.csv'.format(time)
)
ga_results_df.to_csv( path_to_store_result )
print("[MASTER] results stored at:", path_to_store_result, flush=True)
# FINISHED 
