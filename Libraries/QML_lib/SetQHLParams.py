import numpy as np
import random
import pickle
import argparse
# import UserFunctions
import DataBase
import Distrib
import  GrowthRules 

import os
import matplotlib.pyplot as plt
pickle.HIGHEST_PROTOCOL=2
global test_growth_class_implementation
test_growth_class_implementation = True


def MIN_PARAM():
    return -100

def MAX_PARAM():
    return -50

### SET VALUES HERE ####

# set_experimental_data_prior_specific_terms = {
# }

# set_normal_prior_specific_terms = {
# }


# set_uniform_prior_specific_terms = {
# }

# set_true_params = {
# }

### Functions ###
# def create_plot_probe(
#   max_num_qubits = 7, 
#   pickle_file = None,
#   growth_class = None, 
#   **kwargs
# ):
#   kwargs['num_probes'] = 1 # only want a single probe for plotting purposes
#   print("\nPlot probe generated with kwargs:", kwargs, "\n")
    
#   try:
#       plot_probe_dict = growth_class.plot_probe_generator(
#           **kwargs
#       )
#   except:
#       if test_growth_class_implementation == True: raise
#       plot_probe_dict = UserFunctions.get_probe_dict(
#           **kwargs
#       )

#   for k in list(plot_probe_dict.keys()):
#       # replace tuple like key returned, with just dimension. 
#       plot_probe_dict[k[1]] = plot_probe_dict.pop(k)

#   if pickle_file is not None:
#       import pickle
#       pickle.dump(
#           plot_probe_dict,
#           open(pickle_file, 'wb') 
#       )

def create_qhl_params(
    true_op, 
    true_prior, 
    pickle_file=None,
    random_vals=False, 
    growth_generator=None,
    unique_growth_classes=None, 
    all_growth_classes=None, 
    rand_min=None, 
    rand_max=None,
    exp_data=0,
    growth_class = None, 
    plus_probe_for_plot=False,
    true_prior_plot_file=None, 
):
    if rand_min is None:
        rand_min = MIN_PARAM()
    if rand_max is None:
        rand_max = MAX_PARAM()

    terms = DataBase.get_constituent_names_from_name(
        true_op
    )

    latex_terms = []
    for term in terms:
        lt = growth_class.latex_name(
            name = term 
        )
        latex_terms.append(lt)
    true_op_latex = growth_class.latex_name(
        name = true_op, 
    )

    num_terms = len(terms)
    true_params = []
    true_params_dict = {}
    true_params_dict_latex_names = {}

    # sample from wider distribution than initiated for QML
    widen_prior_factor = 2 # should mean true values within 3 sigma of learning distribution
    old_cov_mtx = true_prior.cov
    new_cov_mtx = widen_prior_factor*old_cov_mtx 
    true_prior.__setattr__('cov', new_cov_mtx)
    sampled_list = true_prior.sample()
    try:
        fixed_true_params = growth_class.true_params
    except:
        fixed_true_params = set_true_params

    for i in range(num_terms):
        if random_vals == True:
            print("[setQHL] using random vals")
            true_param = sampled_list[0][i]
        else:
            try:
                term = terms[i]
                true_param = fixed_true_params[term]
            except:
                true_param = sampled_list[0][i]     
        true_params.append(true_param)
        true_params_dict[terms[i]] = true_param
        true_params_dict_latex_names[latex_terms[i]] = true_param

    true_prior.__setattr__('cov', old_cov_mtx)
    try:
      Distrib.plot_prior(
        model_name = true_op_latex, 
        model_name_individual_terms = latex_terms,
        prior = true_prior, 
        plot_file = true_prior_plot_file,
        true_params = true_params_dict_latex_names
      )
    except:
      print("[SetQHLParams]plotting prior failed \n\n\n")
      pass

    # for term in terms:
    #   if random_vals == False:
    #       try:
    #           this_true_param = set_true_params[term]
    #       except:
    #           this_true_param = random.uniform( rand_min, rand_max)
    #   else:
    #       this_true_param = random.uniform( rand_min, rand_max)

    #   true_params.append( this_true_param )
    #   true_params_dict[term] = this_true_param
            
    true_params_info = {
        'params_list' : true_params,
        'params_dict' : true_params_dict,
        'all_growth_classes' : all_growth_classes,
    }
    if exp_data: 
        print("\n\n\n[SetQHL] EXPDATA -- dont store true vals")
        # so as not to plot "true" params for exp data
        true_params_info['params_dict'] = None
        true_params_info['params_list'] = []
        print("true params info:\n",true_params_info)

    true_params_info['true_op'] = true_op
    true_params_info['growth_generator'] = growth_generator
    if pickle_file is not None:
        import pickle
        pickle.dump(
            true_params_info, 
            open(pickle_file, 'wb')
        )

def create_prior(
    true_op, 
    gaussian, # whether to use normal or uniform prior
    exp_data=False,
    pickle_file=None,
    random_vals=False, 
    sigma=1.5,
    rand_min=None, 
    rand_max=None,
    param_mean=None, 
    param_sigma=None,
    results_directory=None, 
    growth_class = None, 
    log_file=None, 
):
#   terms = DataBase.get_constituent_names_from_name(true_op)
    
    if rand_min is None:
        rand_min = MIN_PARAM()
    if rand_max is None:
        rand_max = MAX_PARAM()
    sigma = growth_class.max_param - growth_class.min_param
    set_prior_specific_terms = growth_class.gaussian_prior_means_and_widths 

    specific_terms = {}
    if random_vals is False:
        # only fill this dict in if the user selects NOT 
        # to use random values as parameters
        terms = list(set_prior_specific_terms.keys())
        for term in terms:
            try:
                # print("[SetParams] setting", term, ":", set_prior_specific_terms[term])
                specific_terms[term] = set_prior_specific_terms[term]
            except: 
                # in case term not in set_prior_specific_terms
                val = random.uniform(growth_class.min_param, growth_class.max_param)
                specific_terms[term] = [val, sigma]
   
    true_prior = Distrib.get_prior(
        model_name = true_op, 
        gaussian = True, 
        # param_minimum = rand_min,
        # param_maximum = rand_max,
        param_minimum = growth_class.min_param,
        param_maximum = growth_class.max_param,
        param_normal_mean = param_mean, 
        param_normal_sigma = param_sigma,
        random_mean = False, # if set to true, chooses a random mean between given uniform min/max
        specific_terms = specific_terms, 
        log_file = log_file,
        log_identifier = '[SetQHLParams]'
    )

    prior_data = {
        'specific_terms' : specific_terms, 
        'true_prior' : true_prior
    }

    if pickle_file is not None:
        import pickle
        # print(
        #   "\n\n storing specific terms to", 
        #   pickle_file, 
        #   "\n", prior_data, 
        #   "\n\n"
        # )
        pickle.dump(
            prior_data, 
            open(pickle_file, 'wb')
        )
        # pickle.dump(
        #   specific_terms, 
        #   open(pickle_file, 'wb')
        # )
    return specific_terms, true_prior
            

### Parse arguments from bash
parser = argparse.ArgumentParser(
    description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
  '-true', '--true_params_file', 
  help="File to pickle true params list to.",
  type=str,
  default=None
)
parser.add_argument(
  '-rand_t', '--random_true_params', 
  help="Bool: use random true parameters or those defined in this file.",
  type=int,
  default=0
)

parser.add_argument(
  '-prior', '--prior_file', 
  help="File to pickle prior specific terms to.",
  type=str,
  default=None
)

parser.add_argument(
  '-rand_p', '--random_prior_terms', 
  help="Bool: use random true parameters or those defined in this file.",
  type=int,
  default=0
)

# parser.add_argument(
#   '-op', '--true_op', 
#   help="Operator to generate true params and prior for.",
#   type=str,
#   default='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
# )

parser.add_argument(
  '-exp', '--use_experimental_data', 
  help="Bool: use experimental data or not.",
  type=int,
  default=0
)

parser.add_argument(
  '-ggr', '--growth_generation_rule', 
  help="Generator of new models",
  type=str,
  default=0
)

parser.add_argument(
  '-agr', '--alternative_growth_rules',
  help='Growth rules to form other trees.',
  # type=str,
  action='append',
  default=[],
)

parser.add_argument(
  '-log', '--log_file',
  help='File to log RQ workers.',
  type=str,
  default='qmd.log'
)

parser.add_argument(
  '-dir', '--results_directory',
  help='Relative directory to store results in.',
  type=str,
  default=os.getcwd()
)

parser.add_argument(
  '-probe', '--plot_probe_file', 
  help="File to pickle probes against which to plot expectation values.",
  type=str,
  default=None
)

parser.add_argument(
  '-sp', '--special_probe', 
  help="Special type of probe, e.g. |+>, or ideal (sum of eigenstates).",
  type=str,
  default=None
)

parser.add_argument(
  '-plus', '--force_plus_probe', 
  help="Whether to enforce plots to use |+>^n as probe.",
  type=int,
  default=0
)

parser.add_argument(
  '-g', '--gaussian', 
  help="Whether to use normal (Gaussian) distribution. If False: uniform.",
  type=int,
  default=1
)

parser.add_argument(
  '-min', '--param_min', 
  help="Minimum valid parameter value",
  type=float,
  default=0.0
)
parser.add_argument(
  '-max', '--param_max', 
  help="Maximum valid parameter value",
  type=float,
  default=1.0
)

parser.add_argument(
  '-mean', '--param_mean', 
  help="Default mean of normal distribution for unspecified parameters.",
  type=float,
  default=0.5
)
parser.add_argument(
  '-sigma', '--param_sigma', 
  help="Default sigma of normal distribution for unspecified parameters.",
  type=float,
  default=0.25
)

parser.add_argument(
  '-pnoise', '--probe_noise_level',
  help='Noise level to add to probe for learning',
  type=float,
  default=0.03
)


arguments = parser.parse_args()
random_true_params = bool(arguments.random_true_params)
random_prior = bool(arguments.random_prior_terms)
exp_data = bool(arguments.use_experimental_data)
growth_generation_rule = arguments.growth_generation_rule
log_file = arguments.log_file
results_directory = arguments.results_directory

import ProbeGeneration
growth_class_attributes = {
    'use_experimental_data' : exp_data, 
    'log_file' : log_file
    # 'probe_generator' : ProbeGeneration.restore_dec_13_probe_generation,
    # 'test_growth_class_att' : True
}

growth_class = GrowthRules.get_growth_generator_class(
    growth_generation_rule = growth_generation_rule,
    **growth_class_attributes
    # use_experimental_data = exp_data
)

all_growth_classes = [growth_generation_rule]
alternative_growth_rules = arguments.alternative_growth_rules
all_growth_classes.extend(alternative_growth_rules)
all_growth_classes = list(set(all_growth_classes))

print("[SETQHLPARAMS] growth generation rule:", growth_generation_rule)
print("[SETQHLPARAMS] all growth classes:", all_growth_classes)

unique_growth_classes = {}
for g in all_growth_classes:
    try:
        unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
            growth_generation_rule = g,
            **growth_class_attributes
        )
    except:
        unique_growth_classes[g] = None




true_operator = growth_class.true_operator
plot_probe_file = arguments.plot_probe_file
force_plus_probe = bool(arguments.force_plus_probe)
special_probe = arguments.special_probe
gaussian = bool(arguments.gaussian)
param_min = arguments.param_min
param_max = arguments.param_max
param_mean = arguments.param_mean
param_sigma = arguments.param_sigma
probe_noise_level = arguments.probe_noise_level
# print(
#   "Results directory passed to SetQHL:", 
#   results_directory,
#   "\nLog file:", log_file
# )
true_prior_plot_file = str(
    results_directory  +
    '/prior_true_params.png'
)
### Call functions to create pickle files. 
## TODO check if these are already present?

specific_terms, true_prior = create_prior(
    # true_op = arguments.true_op,
    true_op = true_operator, 
    gaussian = gaussian, 
    pickle_file = arguments.prior_file,
    random_vals = random_prior, 
    rand_min=param_min, 
    rand_max=param_max,
    param_mean = param_mean, 
    param_sigma = param_sigma,
    exp_data=exp_data,
    results_directory=results_directory,
    growth_class = growth_class, 
    log_file=log_file
)

if arguments.true_params_file is not None:
    create_qhl_params(
        # true_op = arguments.true_op, 
        true_op = true_operator,
        true_prior = true_prior,
        pickle_file=arguments.true_params_file,
        growth_generator=growth_generation_rule,
        unique_growth_classes=unique_growth_classes,
        all_growth_classes=all_growth_classes,
        random_vals=random_true_params, 
        rand_min=param_min, 
        rand_max=param_max,
        exp_data=exp_data,
        growth_class = growth_class, 
        true_prior_plot_file=true_prior_plot_file 
    )


"""
if (
    force_plus_probe == True 
    or 
    exp_data == True
):
    force_plus = True
else:
    force_plus = False

"""
###
# Now generate a probe dict to be used by all instances 
# of QMD within this run, when plotting results.
# Store it in the provided argument, plot_probe_dict. 
### 

print("Generating probe dict for plotting")

# TODO 
plot_probe_dict = growth_class.plot_probe_generator(
    true_operator = true_operator, 
    growth_generator = growth_generation_rule,
    experimental_data = exp_data,
    # special_probe = special_probe, 
    # num_probes = 1, 
    noise_level = probe_noise_level, 
)
print("Generated probe dict for plotting")

for k in list(plot_probe_dict.keys()):
    # replace tuple like key returned, with just dimension. 
    plot_probe_dict[k[1]] = plot_probe_dict.pop(k)

if plot_probe_file is not None:
    import pickle
    pickle.dump(
        plot_probe_dict,
        open(plot_probe_file, 'wb') 
    )

