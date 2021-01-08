import sys
import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
import itertools
from matplotlib.lines import Line2D
import qinfer as qi
from scipy import linalg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.markers as mmark
from matplotlib.ticker import PercentFormatter
import hexp

from lfig import LatexFigure
sys.path.append("/home/bf16951/QMD")
import qmla

def load_results(
    results_time,
    results_folder=os.path.abspath("/home/bf16951/thesis/qmla_run_data/"),
    run_id='001'
):

    results_dir = os.path.join(
        results_folder, 
        results_time
    )

    try:
        results_file = os.path.join(results_dir, 'results_{}.p'.format(run_id))
        res = pickle.load(open(results_file, 'rb'))
    except:
        results_file = os.path.join(results_dir, 'results_m1_q{}.p'.format(run_id))
        res = pickle.load(open(results_file, 'rb'))


    true_params = pickle.load(open(os.path.join(results_dir, 'run_info.p'), 'rb')) 
    qmla_class_file = os.path.join(results_dir, 'qmla_class_{}.p'.format(run_id))
    plot_probes = pickle.load(open(os.path.join(results_dir, 'plot_probes.p'), 'rb'))
    true_measurements = pickle.load(open(os.path.join(results_dir, 'system_measurements.p'), 'rb'))
    q = pickle.load(open(qmla_class_file, 'rb'))
    try:
        q2 = pickle.load(open(os.path.join(results_dir, 'qmd_class_002.p'), 'rb'))
    except:
        pass
    es = q.exploration_class
    try:
        combined_datasets = os.path.join(results_dir, 'combined_datasets')
        evaluation_data = pickle.load(open(os.path.join(results_dir, 'evaluation_data.p' ), 'rb'))
        storage = pickle.load(open(os.path.join(results_dir, 'storage_{}.p'.format(run_id)), 'rb'))
        system_probes = pickle.load(open(
            os.path.join(results_dir, 'training_probes', 'system_probes.p'),
            'rb'
        ))
        ga = gr.genetic_algorithm
    except:
        pass

    try:
        # these are only available if analysis has been performed
        champ_info = pickle.load(open(os.path.join(results_dir, 'champion_models',  'champions_info.p' ), 'rb'))
        bf = pd.read_csv(os.path.join(combined_datasets,  'bayes_factors.csv' ))
        fitness_df = pd.read_csv(os.path.join(combined_datasets,  'fitness_df.csv' ))
        combined_results = pd.read_csv(os.path.join(results_dir, 'combined_results.csv'))
        correlations = pd.read_csv(
            os.path.join(combined_datasets, "fitness_correlations.csv")
        )
        fitness_by_f_score = pd.read_csv(
            os.path.join(combined_datasets, 'fitness_by_f_score.csv')
        )
    except:
        pass
    
    results = {
        'qmla_instance' : q, 
        'exploration_strategy' : es, 
        'results_dir' : results_dir,
        'true_measurements' : true_measurements, 
        'true_params' : true_params
    }
    
    return results

import scipy

particle = {
    "FH-hopping-sum_down_1h2_1h3_2h4_3h4_d4" : 0.610602, 
    "FH-hopping-sum_up_1h2_1h3_2h4_3h4_d4" : 0.70319334,
    "FH-onsite-sum_1_2_3_4_d4" : 0.29559276,
}

particle = {
    "FH-hopping-sum_down_1h2_d2" : 0.610602, 
    # "FH-hopping-sum_up_1h2_d2" : 0.70319334,
    # "FH-onsite-sum_1_2_d2" : 0.29559276,
}

# particle = {
#     'pauliSet_1_x_d4' : 0.2
# }


model = sum([particle[t]*qmla.construct_models.compute(t) for t in particle])
print("Model has strides {}".format(model.strides))

t= 4.87027981337 
probe = qmla.shared_functionality.probe_set_generation.random_probe(8)

h = hexp.UnitaryEvolvingMatrix(model, t)
h.expm_hermitian()
# h.expm_hermitian_sparse()