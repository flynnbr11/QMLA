import sys
import os
import numpy as np
import pandas as pd
import time
import pickle
import datetime

from param_sweep_utilities import get_all_configurations, run_genetic_algorithm


ga_results_df = pd.DataFrame() 

# get configurations to cycle over
all_configurations, configuration_df = get_all_configurations(
    log_file = os.path.join(
        os.getcwd(), 'output.log'
    ),
)
print("all_configurations has type {}".format(type(all_configurations)))
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
now = datetime.datetime.now()

result_directory = os.path.join(
    os.getcwd(), 
    'results',
    "{}_{}".format(now.strftime("%b"), now.strftime("%d")),
    "{}_{}".format(now.strftime("%H"), now.strftime("%M"))
)
path_to_store_result = os.path.join(
    result_directory,
    'results.csv'.format(time)
)

try:
    os.makedirs(result_directory)
    print("Directory made", flush=True)
except:
    print("Failed to make dir", flush=True)
    raise

ga_results_df.to_csv( path_to_store_result )
print("[MASTER] results stored at:", path_to_store_result, flush=True)
# FINISHED 
