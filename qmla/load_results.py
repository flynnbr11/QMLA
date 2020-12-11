import os
import sys

import pickle
import pandas as pd

import qmla

p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-2]
qmla_root = os.path.abspath('/'.join(elements))
     

def load_results(
    results_time,
    results_folder_elements=[qmla_root, "launch", "results"], 
    results_folder=None, 
    instance_id='001'
):
    r""" 
    Load results packet from a run of QMLA.

    :param str results_time: time at which the run was started
    :param str results_folder: elements of path to results storage
    """
   
    if results_folder is None:
        results_folder=os.path.abspath(os.path.join(
            *results_folder_elements
        ))

    results_dir = os.path.join(
        results_folder, 
        results_time
    )
    instance_id = '{0:03d}'.format(int(instance_id))

    try:
        results_file = os.path.join(results_dir, 'results_{}.p'.format(instance_id))
        res = pickle.load(open(results_file, 'rb'))
    except:
        results_file = os.path.join(results_dir, 'results_m1_q{}.p'.format(instance_id))
        res = pickle.load(open(results_file, 'rb'))

    try:
        run_info = pickle.load(open(os.path.join(results_dir, 'run_info.p'), 'rb')) 
    except:
        run_info = pickle.load(open(os.path.join(results_dir, 'true_params.p'), 'rb'))  # old runs used this

    qmla_class_file = os.path.join(results_dir, 'qmla_class_{}.p'.format(instance_id))
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
        storage = pickle.load(open(os.path.join(results_dir, 'storage_{}.p'.format(instance_id)), 'rb'))
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
        'run_info' : run_info,
        'storage' : storage
    }
    
    return results