import os
import csv
import pickle
import pandas as pd
pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'collect_results_store_csv',
    'count_model_occurences'
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

