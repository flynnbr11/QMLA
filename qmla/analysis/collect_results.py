import os
import csv
import pickle
import pandas as pd
pickle.HIGHEST_PROTOCOL = 4

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = [
    'collect_results_store_csv',
    'generate_combined_datasets',
    'count_model_occurences',
    # '_generate_combined_datasets'
]

def collect_results_store_csv(
    directory_name,
    results_file_name_start="results",
    results_csv_name="results.csv", 
    csv_name='all_results.csv'
):
    r""" 
    Iteratively opens all files starting with 
    results_file_name_start and ending with .p,
    which are stored at the end of each completed 
    QMLA instance. 
    The results are gathered into a single CSV.

    :param directory_name: the directory to search in for results
    :param results_file_name_start: can distinguish which type of QMLA
        had run; i.e. different when multiple_model_qhl mode is used, 
        than standard QMLA. 
    :param results_csv_name: the name to store the resultant CSV by. 

    :returns collected_results: pandas DataFrame with results from all instances.  
    """
    results_csv = os.path.join(directory_name, results_csv_name)

    pickled_files = []
    for file in os.listdir(directory_name):
        if (
            file.endswith(".p")
            and
            file.startswith(results_file_name_start)
        ):
            pickled_files.append(file)
    filenames = [directory_name + str(f) for f in pickled_files]
    try:
        some_results = pickle.load(open(filenames[0], "rb"))

    except BaseException:
        print("collect_results: Couldn't find results files beginning with ",
              results_file_name_start
              )

        print(
            "directory:", directory_name,
            "\nresults_file_name_start:", results_file_name_start,
            "\nFilenames found:", filenames,
            "\npickled files:", pickled_files,
            "\nlistdir:", os.listdir(directory_name)
        )
        raise

    result_fields = list(some_results.keys())
    with open(results_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_fields)
        writer.writeheader()

        for f in filenames:
            results = pickle.load(open(f, "rb"))
            writer.writerow(results)

    collected_results = pd.read_csv(results_csv)
    return collected_results



# def _generate_combined_datasets(
#     directory_name,
#     results_file_name_start="results",
#     results_csv_name="results.csv", 
# ):
#     r""" DEPRECATED"""

#     pickled_files = []
#     for file in os.listdir(directory_name):
#         # TODO unload storage objects instead??
#         # then could pickle more e.g. pd DataFrames directly
#         if (
#             file.endswith(".p")
#             and
#             file.startswith(results_file_name_start)
#         ):
#             pickled_files.append(file)
#     filenames = [directory_name + str(f) for f in pickled_files]

#     # datasets to store
#     fitness_correlations = pd.DataFrame()

#     for f in filenames:
#         result = pickle.load(open(f, 'rb'))
#         result_id = result['QID']

#         try:
#             correlations = pd.DataFrame(result['GrowthRuleStorageData']['fitness_correlations'])
#             correlations['qmla_id'] = result_id
#             fitness_correlations = fitness_correlations.append(
#                 correlations, 
#                 ignore_index = True
#             )
#         except:
#             raise
#             pass


#     try:
#         fitness_correlations.to_csv(
#             os.path.join(directory_name, 'fitness_method_correlations.csv')
#         )
#         fig = sns.catplot(
#             y  = 'Correlation', 
#             x = 'Method',
#             data = fitness_correlations,
#             kind='box'
#         )
#         fig.savefig(
#             os.path.join(
#                 directory_name, "fitness_f_score_correlations.png"
#             )
#         )
#     except:
#         print("ANALYSIS FAILURE: fitness method  &  score correlations")
#         raise
#         # pass

def generate_combined_datasets(
    directory_name, # where storage results are
    combined_datasets_directory, # where to save datasets
):
    # combined_datasets_directory = os.path.join(
    #     directory_name, 'combined_datasets'
    # )
    if not os.path.exists(combined_datasets_directory):
        try:
            os.makedirs(combined_datasets_directory)
        except:
            pass

    storage_files = []
    for file in os.listdir(directory_name):
        # TODO unload storage objects instead??
        # then could pickle more e.g. pd DataFrames directly
        if (
            file.endswith(".p")
            and
            file.startswith("storage")
        ):
            storage_files.append(file)
    filenames = [
        os.path.join(directory_name, str(f)) for f in storage_files
    ]

    # combined datasets to generate
    bayes_factors = pd.DataFrame()
    fitness_correlations = pd.DataFrame()
    fitness_by_f_score = pd.DataFrame()
    fitness_df = pd.DataFrame()
    all_models_generated = pd.DataFrame()
    misc_gr_data = pd.DataFrame()
    unique_chromosomes = pd.DataFrame()
    lattice_record = pd.DataFrame()
    gene_pool = pd.DataFrame()
    birth_register = pd.DataFrame()
    gen_alg_summary = pd.DataFrame()

    # cycle through files
    for f in filenames:
        storage = pickle.load(open(f, 'rb'))

        bf = storage.bayes_factors_df
        bf['qmla_id'] = storage.qmla_id
        bayes_factors = bayes_factors.append(bf, ignore_index=True)

        try:
            fit_cor = storage.growth_rule_storage.fitness_correlations
            fit_cor['qmla_id'] = storage.qmla_id
            fitness_correlations = fitness_correlations.append(fit_cor, ignore_index=True)
        except:
            pass
        
        try:
            fit_f = storage.growth_rule_storage.fitness_by_f_score
            fit_f['qmla_id'] = storage.qmla_id
            fitness_by_f_score = fitness_by_f_score.append(fit_f, ignore_index=True)
        except:
            pass

        try:
            models_generated = storage.models_generated
            models_generated['qmla_id'] = storage.qmla_id
            all_models_generated = all_models_generated.append(models_generated, ignore_index=True)
        except:
            print("Failed to add generated models.")
            raise
            # pass

        try:
            fit_f = storage.growth_rule_storage.fitness_df
            fit_f['qmla_id'] = storage.qmla_id
            fitness_df = fitness_df.append(fit_f, ignore_index=True)
        except:
            pass                   

        try:
            gp = storage.growth_rule_storage.gene_pool
            gp['qmla_id'] = storage.qmla_id
            gp['time'] = storage.Time
            gene_pool = gene_pool.append(gp, ignore_index=True)
        except:
            pass                   

        try:
            br = storage.growth_rule_storage.birth_register
            br['qmla_id'] = storage.qmla_id
            br['time'] = storage.Time
            birth_register = birth_register.append(br, ignore_index=True)
        except:
            pass                   

        try:
            gs = pd.Series({
                'qmla_id' : storage.qmla_id, 
                'champ_f_score' : storage.Fscore, 
                'num_models' : storage.NumModels, 
                'champ_terms_latex' : storage.ConstituentTerms, 
                'champ_terms' : storage.NameAlphabetical.split('+'), 
                'true_found' : bool(storage.CorrectModel),
                'num_generations' : storage.growth_rule_storage.birth_register.generation.max(),
                'time_taken' : storage.Time
            })
            gen_alg_summary = gen_alg_summary.append(gs, ignore_index=True)
        except:
            pass

        try:
            # NOTE chromosomes cast to integers when written to CSV
            # so they may be shorter than chomosome 
            # and should be recast to chromosome length
            # TODO just store via to_pickle, then read_pickle, and they will work. s
            uc = storage.growth_rule_storage.unique_chromosomes
            uc['qmla_id'] = storage.qmla_id
            uc['true_chromosome'] = storage.growth_rule_storage.true_model_chromosome
            num_terms = uc.num_terms.unique()[0]
            # method to retrieve full chromosome (also will store as float -- do this during application)
            uc['full_chromosome'] = [format( int( str(c), 2),  '0{}b'.format(num_terms)) for c in uc.chromosome ] 
            unique_chromosomes = unique_chromosomes.append(
                uc, ignore_index=True
            )
        except:
            pass

        try:
            instance_lattice = storage.growth_rule_storage.lattice_record
            instance_lattice['qmla_id'] = storage.qmla_id
            instance_lattice['true_model_found'] = storage.TrueModelFound
            lattice_record = lattice_record.append(instance_lattice, ignore_index=True)
        except:
            pass                   

    # Store datasets and add their name to the list
    datasets_generated = []

    bayes_factors.to_csv(os.path.join(combined_datasets_directory, 'bayes_factors.csv'))
    datasets_generated.append('bayes_factors')

    try:
        fitness_correlations.to_csv(os.path.join(
            combined_datasets_directory, 'fitness_correlations.csv')
        )
        datasets_generated.append('fitness_correlations')
    except:
        pass

    try:
        fitness_by_f_score.to_csv(os.path.join(
            combined_datasets_directory, 'fitness_by_f_score.csv')
        )
        datasets_generated.append('fitness_by_f_score')
    except:
        pass

    try:
        fitness_df.to_csv(
            os.path.join( combined_datasets_directory, 'fitness_df.csv')
        )
        datasets_generated.append('fitness_df')
    except:
        pass

    try:
        all_models_generated.to_csv(
            os.path.join( combined_datasets_directory, 'models_generated.csv')
        )
        datasets_generated.append('all_models_generated')
    except:
        pass

    try:
        growth_rule_data.to_csv(
            os.path.join( combined_datasets_directory, 'growth_rule_data.csv')
        )
        datasets_generated.append('growth_rule_data')
    except:
        pass
    
    try:
        birth_register.to_pickle(
            os.path.join( combined_datasets_directory, 'birth_register.p')
        )
        datasets_generated.append('birth_register')
    except:
        pass

    try:
        gen_alg_summary.to_pickle(
            os.path.join( combined_datasets_directory, 'gen_alg_summary.p')
        )
        datasets_generated.append('gen_alg_summary')
    except:
        pass

    try:
        gene_pool.to_pickle(
            os.path.join( combined_datasets_directory, 'gene_pool.p')
        )
        datasets_generated.append('gene_pool')
    except:
        pass

    try:
        unique_chromosomes.to_csv(
            os.path.join( combined_datasets_directory, 'unique_chromosomes.csv'),
        )
        datasets_generated.append('unique_chromosomes')
    except:
        raise
        # pass
    
    try:
        lattice_record.to_csv(
            os.path.join( combined_datasets_directory, 'lattice_record.csv'),
        )
        datasets_generated.append('lattice_record')
    except:
        raise

    # Gather together and return
    # combined_data = {
    #     'results_directory' : combined_datasets_directory,  
    #     'datasets_generated' : datasets_generated      
    # }
    return datasets_generated


def count_model_occurences(
    latex_map,
    true_model_latex,
    save_counts_dict=None,
    save_to_file=None
):
    r"""
    Plots each model considered within this run against the number 
    of times that model was considered. 
    Quite inefficient and usually not very informative/useful, 
    so turned off by default. 

    :param latex_map: path to txt file listing model strings and 
        their corresponding latex representation, provided by QMLA.
    :param true_model_latex: latex representation of true model,
        required so the plot can highlight that model in a different colour.
    :param save_counts_dict: if not None, the path to save the results to. 
    :param save_to_file: if not None, the path to save the resultant PNG to. 
    """
    f = open(latex_map, 'r')
    l = str(f.read())
    terms = l.split("',")

    # for t in ["(", ")", "'", " "]:
    for t in ["'", " "]:
        terms = [a.replace(t, '') for a in terms]

    sep_terms = []
    for t in terms:
        sep_terms.extend(t.split("\n"))

    unique_models = list(set([s for s in sep_terms if "$" in s]))
    counts = {}
    for ln in unique_models:
        counts[ln] = sep_terms.count(ln)
    unique_models = sorted(unique_models)
    model_counts = [counts[m] for m in unique_models]
    unique_models = [
        a.replace("\\\\", "\\")
        for a in unique_models
    ]  # in case some models have too many slashes.
    max_count = max(model_counts)
    integer_ticks = list(range(max_count + 1))
    colours = ['blue' for m in unique_models]
    unique_models = [u[:-1] for u in unique_models if u[-1] == ')']
    true_model_latex = true_model_latex.replace(' ', '')
    if true_model_latex in unique_models:
        true_idx = unique_models.index(true_model_latex)
        colours[true_idx] = 'green'

    fig, ax = plt.subplots(
        figsize=(
            max(max_count * 2, 5),
            len(unique_models) / 2)
    )
    ax.plot(kind='barh')
    ax.barh(
        unique_models,
        model_counts,
        color=colours
    )
    ax.set_xticks(integer_ticks)
    ax.set_title('# times each model generated')
    ax.set_xlabel('# occurences')
    ax.tick_params(
        top=True,
        direction='in'
    )
    if save_counts_dict is not None:
        import pickle
        pickle.dump(
            counts,
            open(
                save_counts_dict,
                'wb'
            )
        )

    try:
        if save_to_file is not None:
            plt.savefig(
                save_to_file,
                bbox_inches='tight'
            )
    except BaseException:
        print(
            "[collect_results - count model occurences] couldn't save plot to file",
            save_to_file

        )
        raise
