import random
import pickle
import argparse
import UserFunctions
import DataBase


def MIN_PARAM():
	return -100

def MAX_PARAM():
	return -50

### SET VALUES HERE ####

set_experimental_data_prior_specific_terms = {
	# 'xTi' : [4.0, 1.5],
	# 'yTi' : [4.0, 1.5],
	# 'zTi' : [4.0, 1.5],
	# 'xTx' : [4.0, 1.5],
	# 'yTy' : [4.0, 1.5],
	# 'zTz' : [4.0, 1.5],
	# 'xTy' : [4.0, 1.5],
	# 'xTz' : [4.0, 1.5],
	# 'yTz' : [4.0, 1.5],
	# 'xTi': [5.0, 2.0], 
	# 'xTy': [3.0, 2.0], 
	# 'zTi': [-5.0, 2.0], 
	# 'xTx': [-5.0, 2.0], 
	# 'zTz': [-5.0, 2.0], 
	# 'z': [1.0, 0.5], 
	# 'xTz': [3.0, 2.0], 
	# 'yTi': [5.0, 2.0], 
	# 'x': [1.0, 0.5], 
	# 'yTz': [3.0, 2.0], 
}

set_normal_prior_specific_terms = {
	# 'x' : [5.0, 2.0],
	# 'z' : [5.0, 2.0],
	# 'y' : [5.0, 2.0]
	# 'xTi' : [4.0, 1.5],
	# 'yTi' : [4.0, 1.5],
	# 'zTi' : [4.0, 1.5],
	# 'xTx' : [4.0, 1.5],
	# 'yTy' : [4.0, 1.5],
	# 'zTz' : [4.0, 1.5],
	# 'xTy' : [4.0, 1.5],
	# 'xTz' : [4.0, 1.5],
	# 'yTz' : [4.0, 1.5],
	# 'xTi' : [0.0, 1.5],
	# 'yTi' : [0.0, 1.5],
	# 'zTi' : [0.0, 1.5],
	# 'xTx' : [0.0, 1.5],
	# 'yTy' : [0.0, 1.5],
	# 'zTz' : [0.0, 1.5],
	# 'xTy' : [0.0, 1.5],
	# 'xTz' : [0.0, 1.5],
	# 'yTz' : [0.0, 1.5],
}

set_uniform_prior_specific_terms = {
}

set_true_params = {
	# 'a' : 0.44, 
	# 's' : 0.44,
	# 'h_1_2_d3' : 0.6428,
	# 'h_1_3_d3' : 0.9098, 
	# 'h_2_3_d3' : 0.9051
	# 'xTi' : 0.7538085315,
	# 'yTi' : 8.4442484,
	# 'zTi' : 5.713337168736279,	

	# From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/average_priors
	# 'xTi': -0.57330033068583841, 
	# 'xTx': 5.1275507795542854, 
	# 'yTi': 6.2520709139461959, 
	# 'yTy': 3.3194961601144444, 
	# 'zTi': 2.7674066823006291,
	# 'zTz': 4.2811650247256994,

	# From  BC /Jan_29/17_58/results_007
	'xTi': -0.15493082869796826, 
	'xTx': -1.2914966893450925, 
	# 'xTx': -0.24, # seems to be degeneracy on these values for xTx
	'yTi': 5.996392378378391, 
	'yTy': 0.32964763301295075, 
	'zTi': -0.21973259068334058, 
	'zTz': -0.0083048687787120271,
	# From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/results_049
	# 'xTi': -0.98288958683093952, 
	# 'yTi': 6.4842202054983122, 
	# 'zTi': 0.96477790489201143, 
	# 'xTx': 6.7232235286284681, 
	# 'yTy': 2.7377867056770397, 
	# 'zTz': 1.6034234519563935,	# 'xTi': 0.20535428954933724, 
}


### Functions ###


def create_plot_probe(
	max_num_qubits = 7, 
	pickle_file = None,
	# plus_probe_for_plot = True,
	# experimental_data=True, 
	# growth_generator=None,
	**kwargs
):
	# import ExpectationValues
	import numpy as np
	# kwargs['test_fill'] = 98
	# print("[createPlotProbe] kwargs", kwargs)
	# print("[createPlotProbe] exp data:", experimental_data )
	# print("[createPlotProbe] ggr:",growth_generator )
	

	kwargs['num_probes'] = 1
	print("\nPlot probe generated with kwargs:", kwargs, "\n")
	plot_probe_dict = UserFunctions.get_probe_dict(
		**kwargs
	)
	for k in list(plot_probe_dict.keys()):
		# replace tuple like key returned, with just dimension. 
	    plot_probe_dict[k[1]] = plot_probe_dict.pop(k)

	if pickle_file is not None:
		import pickle
		pickle.dump(
			plot_probe_dict,
			open(pickle_file, 'wb') 
		)

def create_qhl_params(
	true_op, 
	pickle_file=None,
	random_vals=False, 
	growth_generator=None,
	rand_min=None, 
	rand_max=None,
	exp_data=0,
	plus_probe_for_plot=False
):
	if rand_min is None:
		rand_min = MIN_PARAM()
	if rand_max is None:
		rand_max = MAX_PARAM()

	terms = DataBase.get_constituent_names_from_name(true_op)
	true_params = []

	true_params_dict = {}

	for term in terms:
		try:
			this_true_param = set_true_params[term]
		except:
			this_true_param = random.uniform( rand_min, rand_max)

		true_params.append( this_true_param )
		true_params_dict[term] = this_true_param
			
	true_params_info = {
		'params_list' : true_params,
		'params_dict' : true_params_dict
	}
	if exp_data: 
		# so as not to plot "true" params for exp data
		true_params_info['params_dict'] = None

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
):
#	terms = DataBase.get_constituent_names_from_name(true_op)
	
	if rand_min is None:
		rand_min = MIN_PARAM()
	if rand_max is None:
		rand_max = MAX_PARAM()

	# print("[SetParams] CREATE PRIOR:")
	# print("[SetParams] random vals:", random_vals, )
	# print("[SetParams] min/max", rand_min, rand_max)
	# print("[SetParams] Gaussian:", gaussian)

	# TODO should there be an experimental uniform option?
	if exp_data is True:
		set_prior_specific_terms = set_experimental_data_prior_specific_terms
	elif gaussian ==  True:
		set_prior_specific_terms = set_normal_prior_specific_terms
	else:
		set_prior_specific_terms = set_uniform_prior_specific_terms

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
				val = random.uniform(rand_min, rand_max)
				specific_terms[term] = [val, sigma]
	# print("[SetParams] specific terms:", specific_terms)


	# terms = list(set_prior_specific_terms.keys())
	# if random_vals is True:
	# 	for term in terms:
	# 		val = random.uniform(rand_min, rand_max)
	# 		specific_terms[term] = [val, sigma]
	# else:
	# 	for term in terms:
	# 		try:
	# 			specific_terms[term] = set_prior_specific_terms[term]
	# 		except: 
	# 			# in case term not in set_prior_specific_terms
	# 			val = random.uniform(rand_min, rand_max)
	# 			specific_terms[term] = [val, sigma]
	# print("PRIOR:", specific_terms)
	if pickle_file is not None:
		import pickle
		pickle.dump(
			specific_terms, 
			open(pickle_file, 'wb')
		)
			
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

parser.add_argument(
  '-op', '--true_op', 
  help="Operator to generate true params and prior for.",
  type=str,
  default='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
)

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

arguments = parser.parse_args()
random_true_params = bool(arguments.random_true_params)
random_prior = bool(arguments.random_prior_terms)
exp_data = bool(arguments.use_experimental_data)
growth_generation_rule = arguments.growth_generation_rule
true_operator = UserFunctions.default_true_operators_by_generator[
	growth_generation_rule
]
plot_probe_file = arguments.plot_probe_file
force_plus_probe = bool(arguments.force_plus_probe)
special_probe = arguments.special_probe
gaussian = bool(arguments.gaussian)
param_min = arguments.param_min
param_max = arguments.param_max
param_mean = arguments.param_mean
param_sigma = arguments.param_sigma

### Call functions to create pickle files. 
## TODO check if these are already present?
if arguments.true_params_file is not None:
	create_qhl_params(
		# true_op = arguments.true_op, 
		true_op = true_operator,
		pickle_file=arguments.true_params_file,
		growth_generator=growth_generation_rule,
		random_vals=random_true_params, 
		rand_min=param_min, 
		rand_max=param_max,
		exp_data=exp_data
	)

if arguments.prior_file is not None:
	create_prior(
		# true_op = arguments.true_op,
		true_op = true_operator, 
		gaussian = gaussian, 
		pickle_file = arguments.prior_file,
		random_vals = random_prior, 
		rand_min=param_min, 
		rand_max=param_max,
		exp_data=exp_data
	)

if (
	force_plus_probe == True 
	or 
	exp_data == True
):
	force_plus = True
else:
	force_plus = False


kwargs_for_plot_probe_creation = {
	'true_operator' : true_operator, 
	'growth_generator' : growth_generation_rule,
	'experimental_data' : exp_data,
	'special_probe' : special_probe, 
}

create_plot_probe(
	pickle_file = plot_probe_file,
	max_num_qubits = 10,
	**kwargs_for_plot_probe_creation
)