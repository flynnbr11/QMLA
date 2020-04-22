import qinfer
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import qmla.database_framework as database_framework
import qmla.logging

__all__ = [
    'gaussian_prior'
]

def log_print(
    to_print_list,
    log_file, 
    log_identifier='Distributions'
):
    r"""Writng to unique QMLA instance log."""
    qmla.logging.print_to_log(
        to_print_list = to_print_list, 
        log_file = log_file, 
        log_identifier = log_identifier
    )


def gaussian_prior(
    model_name,
    param_minimum=0,
    param_maximum=1,
    default_sigma=None,
    random_mean=False,  # if set to true, chooses a random mean between given uniform min/max
    prior_specific_terms={},
    log_file='qmd.log',
    log_identifier=None,
    **kwargs
):
    """
    Genearates a QInfer Gaussian distribution .

    Given a model_name, deteremines the number of terms in the model, N.
    Generates a multivariate distribution with N dimensions. 
    This is then used as the initial prior, which QHL uses to learn the 
    model parameters. 
    By default, each parameter's mean is the average of param_min and param_max, 
    with sigma = mean/4. This can be changed by specifying prior_specific_terms: 
        individual parameter's means/sigmas can be given. 

    :param str model_name: Unique string representing a model.
    :param float param_minimum: Lower bound for distribution.
    :param float param_maximum: Upper bound for distribution.
    :param float default_sigma: Width of distribution desired. If None, 
        defaults to 0.25 * (param_max - param_min).
    :param dict prior_specific_terms: Individual parameter mean and sigma
        to enforce in the distribution. 
    :param str log_file: Path of the log file for logging errors.
    :param str log_identifier: Unique identifying sting for logging.
    :return QInfer.Distribution dist: distribution to be used as prior for parameter learning 
        of the named model.
    """

    log_print(
        [
            "Getting prior for model:", model_name,
            "Specific terms:", prior_specific_terms,
        ],
        log_file,
        log_identifier
    )
    individual_terms = database_framework.get_constituent_names_from_name(
        model_name
    )
    num_terms = len(individual_terms)
    available_specific_terms = list(prior_specific_terms.keys())
    means = []
    sigmas = []
    default_mean = np.mean([param_minimum, param_maximum])
    # TODO reconsider how default sigma is generated
    # default_sigma = default_mean/2 # TODO is this safe?
    if default_sigma is None:
        default_sigma = (param_maximum - param_minimum) / 4
    for term in individual_terms:
        if term in available_specific_terms:
            means.append(prior_specific_terms[term][0])
            sigmas.append(prior_specific_terms[term][1])
        else:
            if random_mean:
                rand_mean = random.uniform(
                    param_minimum,
                    param_maximum
                )
                means.append(rand_mean)
            else:
                means.append(default_mean)
            sigmas.append(default_sigma)

    means = np.array(means)
    sigmas = np.array(sigmas)
    cov_mtx = np.diag(sigmas**2)

    dist = qinfer.MultivariateNormalDistribution(
        means,
        cov_mtx
    )
    return dist


def plot_prior(
    model_name,
    model_name_individual_terms,
    prior,
    plot_file,
    true_model_terms_params=None,
):
    r"""
    Plots the given distribution to the given file path.

    :param model_name: 
    :type model_name: str
    :param model_name_individual_terms: List of latex terms for 
        all terms in the model.
    :type model_name_individual_terms: list
    :param prior: distribution to plot
    :type prior: Qinfer.Distribution
    :param plot_file: path to save plot of distribution
    :type plot_file: str
    :param true_model_terms_params: true values of the parameters
        to include in plot
    :type true_model_terms_params: dict
    """
    
    from itertools import cycle
    from matplotlib import cm
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    samples = prior.sample(int(1e5))
    num_params = np.shape(samples)[1]
    ncols = int(np.ceil(np.sqrt(num_params)))
    nrows = int(np.ceil(num_params / ncols))

    fig, axes = plt.subplots(
        figsize=(10, 7),
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
    )
    row = 0
    col = 0
    axes_so_far = 0

    cm_subsection = np.linspace(0, 0.8, num_params)
    colours = [cm.viridis(x) for x in cm_subsection]
    include_legend = False
    for i in range(num_params):

        ax = axes[row, col]
        axes_so_far += 1
        col += 1
        if col == ncols:
            col = 0
            row += 1

        this_param_samples = samples[:, i]
        this_param_mean = np.mean(this_param_samples)
        this_param_dev = np.std(this_param_samples)
        this_param_colour = colours[i % len(colours)]
        latex_term = model_name_individual_terms[i]
        param_label = str(
            latex_term +
            '\n({} $\pm$ {})'.format(
                np.round(this_param_mean, 2),
                np.round(this_param_dev, 2)
            )
        )
        spacing = np.linspace(min(this_param_samples), max(this_param_samples))
        distribution = norm.pdf(spacing, this_param_mean, this_param_dev)
        ls = next(linecycler)
        ax.hist(
            this_param_samples,
            histtype='step',
            fill=False,
            density=True,
            # label=param_label,
            color=this_param_colour
        )

        if true_model_terms_params is not None:
            try:
                true_param = true_model_terms_params[latex_term]
                ax.axvline(
                    true_param,
                    color=this_param_colour,
                    alpha=1,
                    label='True'
                    # linestyle = ls
                )
                include_legend = True
            except BaseException:
                pass  # i.e. this parameter not in true params
        ax.set_title(param_label)
        if include_legend == True:
            ax.legend()

    # plt.legend()
    fig.suptitle('Initial prior for true model')
    fig.subplots_adjust(
        # top = 0.99,
        # bottom=0.01,
        hspace=0.3,
        wspace=0.4
    )
    fig.savefig(plot_file)
    plt.clf()
