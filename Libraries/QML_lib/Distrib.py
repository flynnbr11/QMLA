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
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time

def log_print(
    to_print_list, 
    log_file, 
    log_identifier=None
):
    if log_identifier is None:
        log_identifier='[Distrib]'
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)
    identifier = str(str(time_seconds()) +
            " [" + log_identifier +"]"
        )

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file)


def get_prior(
    model_name, 
    gaussian = True, 
    param_minimum = 0,
    param_maximum = 1,
    param_normal_mean = 0.5, 
    param_normal_sigma = 0.25,
    random_mean = False, # if set to true, chooses a random mean between given uniform min/max
    specific_terms = None, 
    log_file = 'qmd.log',
    log_identifier = None
):

    log_print(
        [
        "Getting prior for model:", model_name,
        "Specific terms:", specific_terms
        ],
        log_file, 
        log_identifier
    )
    individual_terms = DataBase.get_constituent_names_from_name(
        model_name
    )
    num_terms = len(individual_terms)
    available_specific_terms = list(specific_terms.keys())
    if gaussian==False:
        min_max = np.empty([num_terms, 2])
        for i in range(num_terms):
            if individual_terms[i] in available_specific_terms:
                min_max[i] = specific_terms[individual_terms[i]]
            else:
                min_max[i] = [ param_minimum , param_maximum]
        dist = qinfer.UniformDistribution(min_max)
        samples = dist.sample(10)
        log_print(
            [
            "Uniform Prior",
            "\nterms:", individual_terms, 
            "\nCorresponding Min/Max:", 
            min_max,
            "\nSamples:", samples
            ],
            log_file = log_file, 
            log_identifier = log_identifier
        )

        return dist
    
    else:
        means = []
        sigmas = []
        # print("min/max:", param_minimum, param_maximum)
        default_mean = np.mean([param_minimum, param_maximum])
        # TODO reconsider how default sigma is generated
        # default_sigma = (param_maximum - param_minimum)/4 # TODO is this safe?        
        # default_sigma = default_mean/4 # TODO is this safe?        
        default_sigma = default_mean/2 # TODO is this safe?        
        for term in individual_terms:
            if term in available_specific_terms:
                means.append(specific_terms[term][0])
                sigmas.append(specific_terms[term][1])
            else:
                if random_mean:
                    rand_mean = random.uniform(
                        param_minimum, 
                        param_maximum
                    )
                    print("rand mean:", rand_mean)
                    means.append(rand_mean)
                else:
                    # means.append(param_normal_mean)
                    means.append(default_mean)
                # sigmas.append(param_normal_sigma)
                sigmas.append(default_sigma)
                
        means = np.array(means)
        sigmas = np.array(sigmas)
        cov_mtx = np.diag(sigmas**2)
        
        dist = qinfer.MultivariateNormalDistribution(
            means, 
            cov_mtx
        )
        samples = dist.sample(10)
        log_print(
            [
            "Normal Prior",
            "\nMeans:", 
            means,
            "\nCov mtx:",
            cov_mtx,
            "\nSamples:", samples
            ],
            log_file = log_file, 
            log_identifier = log_identifier
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
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    samples = prior.sample(int(1e5))
    num_params = np.shape(samples)[1]
    ncols = int(np.ceil(np.sqrt(num_params)))
    nrows = int(np.ceil(num_params/ncols))

    fig, axes = plt.subplots(
        figsize = (10, 7), 
        nrows=nrows, 
        ncols=ncols,
        squeeze=False,
    )
    row = 0
    col = 0
    axes_so_far = 0

    cm_subsection = np.linspace(0,0.8,num_params)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [ cm.viridis(x) for x in cm_subsection ]

    for i in range(num_params):
        ax = axes[row, col]
        axes_so_far += 1
        col += 1
        if col == ncols:
            col=0
            row+=1

        this_param_samples = samples[:, i]
        this_param_mean = np.mean(this_param_samples)
        this_param_dev = np.std(this_param_samples)
        this_param_colour = colours[i%len(colours)]
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
            except:
                pass # i.e. this parameter not in true params
        ax.set_title(param_label)
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
    print("[Distrib - plot prior] fig saved")
    plt.clf()


def old_plot_prior(
    model_name, 
    model_name_individual_terms,
    prior,
    plot_file,
    true_params=None, 
):
    from itertools import cycle
    from matplotlib import cm
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    samples = prior.sample(int(1e5))
    num_params = np.shape(samples)[1]

    cm_subsection = np.linspace(0,0.8,num_params)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [ cm.viridis(x) for x in cm_subsection ]

    for i in range(num_params):
        this_param_samples = samples[:, i]
        this_param_mean = np.mean(this_param_samples)
        this_param_dev = np.std(this_param_samples)
        this_param_colour = colours[i%len(colours)]
        latex_term = model_name_individual_terms[i]
        param_label = str(
            latex_term + 
            '  ({} $\pm$ {})'.format(
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
        plt.hist(
            this_param_samples, 
            histtype='step', 
            fill=False,
            label=param_label
        )


        if true_params is not None:
            try:
                true_param = true_params[latex_term]
                plt.axvline(
                    true_param, 
                    color=this_param_colour,
                    alpha=1, 
                    # linestyle = ls
                )
            except:
                pass # i.e. this parameter not in true params


    plt.legend()
    plt.title('Prior for {}'.format(model_name))
    plt.savefig(plot_file)
    plt.clf()





def Gaussian(x, mean = 0., sigma = 1.):
    """
    returns a 1D Gaussian distribution from the input vector of positions x
    """
    return norm.pdf(x, loc = mean, scale = sigma)

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
        posterior_fits.append( curve_fit(Gaussian, post_marginal[0], post_marginal[1], p0=p0)[0]  )
        
    posterior = qinfer.MultivariateNormalDistribution(np.array(posterior_fits)[:,0], np.diag(np.array(posterior_fits)[:,1])**2)
    
    return posterior   


#Function which generate a distribution of multiple uniformly [0,1] distributed values of length NumMulti
#NumMulti usually is the number of uniform distributions we want to sample from dimultaneously, 
#usually chosen as len(OpList), where OpList is the list of operators of the Hamiltonian 
#in the model under consideration

# def MultiVariateUniformDistribution(
#     NumMulti, 
#     DistroLimits=np.array([[-0.5,1.5]]) 
# ):
#     DistroBoundsList = np.repeat(
#         DistroLimits, 
#         [NumMulti], 
#         axis=0
#     )
#     DistroList = list(
#         map(lambda DistroBoundsList: qi.UniformDistribution(DistroBoundsList), DistroBoundsList)
#     )
#     OutputDistribution = qi.ProductDistribution(DistroList)
#     return(OutputDistribution)


# def MultiVariateNormalDistributionNocov(
#     NumMulti, 
#     mean=None, 
#     sigmas=None
# ):
#     if mean is None:
#         mean = np.repeat(0.5, NumMulti)
#     if sigmas is None:
#         sigmas = np.repeat(0.1, NumMulti)
        
#     cov_matrix = np.diag(sigmas**2)
#     OutputDistribution = qi.MultivariateNormalDistribution(mean, cov_matrix)
#     return(OutputDistribution)
    
    
    
# # Specific distributions for Ising type model development. 

# def means_sigmas_ising_term(term, specific_terms={}, 
#     rotation_mean=0.5, rotation_sigma=2, 
#     hyperfine_mean=2.5, hyperfine_sigma=0.5,
#     transverse_mean=0.5, transverse_sigma=1.0,
#     default_mean=0.5, default_sigma=0.5
# ):
#     """
#     Get means and sigmas for models in Ising type configurations
#         ie only valid for 2 qubit systems following Ising convention. 
#         to provide specific values of mean/sigma for terms, 
#         pass {term:[mean, sigma]}, eg: specific_terms = {'xTi' : [0.5, 0.2]}.

#     """

#     num_qubits = DataBase.get_num_qubits(term)
#     plus_string = ''
#     for i in range(num_qubits):
#         plus_string+='P'

#     # individual_terms = term.split('PP')
#     individual_terms = term.split(plus_string)
#     num_params = len(individual_terms)
#     means = []
#     sigmas = []

#     rotation_terms = ['xTi', 'yTi', 'zTi']
#     hyperfine_terms = ['xTx', 'yTy', 'zTz']
#     transverse_terms = ['xTy', 'xTz', 'yTz']
    
#     for k in individual_terms:
#         if k in specific_terms:
#             mean = specific_terms[k][0]
#             sigma = specific_terms[k][1]
#             means.append(mean)
#             sigmas.append(sigma)
#         elif k in rotation_terms:
#             means.append(rotation_mean)
#             sigmas.append(rotation_sigma)
#         elif k in hyperfine_terms:
#             means.append(hyperfine_mean)
#             sigmas.append(hyperfine_sigma)
#         elif k in transverse_terms:
#             means.append(transverse_mean)
#             sigmas.append(transverse_sigma)
#         else:
#             means.append(default_mean)
#             sigmas.append(default_sigma)


#     return num_params, np.array(means), np.array(sigmas)

# def normal_distribution_ising(
#     term, 
#     specific_terms={}
# ):
#     num_params, means, sigmas = means_sigmas_ising_term(
#         term = term, 
#         specific_terms = specific_terms
#     )
#     cov_matrix = np.diag(sigmas**2)
    
#     dist = qi.MultivariateNormalDistribution(
#         means, 
#         cov_matrix
#     )
#     print("[Distrib] distribution generated:", dist)   
#     print("[Normal] Given term ", term,
#            ", generated \nmeans:\n", means, "\nsigmas:\n", sigmas
#     )
#     return dist
    
# def uniform_distribution_ising(
#     term, specific_terms={}, 
#     lower_rotation=-0.5, upper_rotation=1.5,
#     lower_hyperfine=-0.5, upper_hyperfine=1.5, 
#     lower_transverse=-0.5, upper_transverse=1.5
# ):

#     rotation = [lower_rotation, upper_rotation]
#     hyperfine = [lower_hyperfine, upper_hyperfine]
#     transverse = [lower_transverse, upper_transverse]
    
#     single_qubit_terms = ['x', 'y', 'z']
#     rotation_terms = ['xTi', 'yTi', 'zTi']
#     hyperfine_terms = ['xTx', 'yTy', 'zTz']
#     transverse_terms = ['xTy', 'xTz', 'yTz']
    
#     if term in single_qubit_terms:
#         # For use in QHL tests, possibly can be removed later. -BF
#         limits = np.array([[-0.5, 1.5]])
#         DistroList = list(map(
#             lambda limits: qi.UniformDistribution(limits), limits)
#         ) 
#         dist = qi.ProductDistribution(DistroList)
#         return dist
    
#     individual_terms = term.split('PP')
#     limits = []

#     for k in individual_terms:
#         if k in specific_terms:
#             limits.append(specific_terms[k])
#         elif k in rotation_terms:
#             limits.append(rotation)
#         elif k in hyperfine_terms:
#             limits.append(hyperfine)
#         elif k in transverse_terms:
#             limits.append(transverse)
#         else:
#             print("Term", k, "not recongised as rotation, hyperfine or transverse") 

#     limits = np.array(limits)
#     DistroList = list(map(
#         lambda limits: qi.UniformDistribution(limits), limits)
#     ) 
#     dist = qi.ProductDistribution(DistroList)

# #    print("[Uniform] Given term ", term, ", generated limits:\n", limits)
#     return dist
