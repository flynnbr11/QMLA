import qinfer
import random
import numpy as np
import DataBase
import matplotlib.pyplot as plt


from scipy.stats import norm
from scipy.optimize import curve_fit


def time_seconds():
    # return time in h:m:s format for logging.
    import datetime
    now = datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour) + ':' + str(minute) + ':' + str(second))
    return time


def log_print(
    to_print_list,
    log_file,
    log_identifier=None
):
    if log_identifier is None:
        log_identifier = '[Distrib]'
    if type(to_print_list) != list:
        to_print_list = list(to_print_list)
    identifier = str(str(time_seconds()) +
                     " [" + log_identifier + "]"
                     )

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file)


def gaussian_prior(
    model_name,
    param_minimum=0,
    param_maximum=1,
    default_sigma=None,
    random_mean=False,  # if set to true, chooses a random mean between given uniform min/max
    prior_specific_terms=None,
    log_file='qmd.log',
    log_identifier=None,
    **kwargs
):

    log_print(
        [
            "Getting prior for model:", model_name,
            "Specific terms:", prior_specific_terms,
        ],
        log_file,
        log_identifier
    )
    individual_terms = DataBase.get_constituent_names_from_name(
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
    true_params=None,
):
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
    #        colours = [ cm.magma(x) for x in cm_subsection ]
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
        # plt.plot(
        #     spacing,
        #     distribution,
        #     label=param_label,
        #     linestyle=ls
        # )
        ax.hist(
            this_param_samples,
            histtype='step',
            fill=False,
            density=True,
            # label=param_label,
            color=this_param_colour
        )

        if true_params is not None:
            try:
                true_param = true_params[latex_term]
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
    fig.suptitle('Initial prior for {}'.format(model_name))
    fig.subplots_adjust(
        # top = 0.99,
        # bottom=0.01,
        hspace=0.3,
        wspace=0.4
    )
    fig.savefig(plot_file)
    plt.clf()


def Gaussian(x, mean=0., sigma=1.):
    """
    returns a 1D Gaussian distribution from the input vector of positions x
    """
    return norm.pdf(x, loc=mean, scale=sigma)


def get_posterior_fromMarginals(
    all_post_marginals
):
    """
    from an input list of posterior marginals from qinfer.update.posterior_marginal
    returns a qinfer posterior MVNormal distribution (as such it already deals with multiple parameters)
    """

    posterior_fits = []

    for idx_param in range(len(all_post_marginals)):

        post_marginal = all_post_marginals[idx_param]
        p0 = [np.mean(post_marginal[0]), np.std(post_marginal[0])]
        posterior_fits.append(
            curve_fit(
                Gaussian,
                post_marginal[0],
                post_marginal[1],
                p0=p0)[0])

    posterior = qinfer.MultivariateNormalDistribution(
        np.array(posterior_fits)[
            :, 0], np.diag(
            np.array(posterior_fits)[
                :, 1])**2)

    return posterior
