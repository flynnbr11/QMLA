import random
import pickle
import argparse

import DataBase

### SET VALUES HERE ###
set_prior_specific_terms = {
	'xTi' : [0.0, 1.0], # TODO Broaden, testing with small dist
	'yTi' : [0.0, 1.0],
	'zTi' : [0.0, 1.0],
	'xTx' : [0.0, 1.0], # true value 2.7
	'yTy' : [0.0, 1.0], # true value 2.7
	'zTz' : [0.0, 1.0], # true value 2.14
	'xTy' : [0.0, 1.0],
	'xTz' : [0.0, 1.0],
	'yTz' : [0.0, 1.0],
}

set_true_params = {
	'xTi' : 1.0,
	'yTi' : 1.2,
	'zTi' : 1.8,
	'xTx' : 1.3,
	'yTy' : 1.4,
	'zTz' : 1.5,
	'xTy' : 1.6,
	'xTz' : 1.7,
	'yTz' : 1.8
}


### Functions ###
def create_qhl_params(
	true_op, 
	pickle_file=None,
	random_vals=False, 
	rand_min=-3, 
	rand_max=3,
	exp_data=0
):
	terms = DataBase.get_constituent_names_from_name(true_op)
	true_params = []

	true_params_dict = {}

	if random_vals is True:
		for term in terms:
			val = random.uniform( rand_min, rand_max) 
			true_params.append( val )
			true_params_dict[term] = val

	else:
		for term in terms:
			true_params.append( set_true_params[term] )
			true_params_dict[term] = set_true_params[term]
			
	true_params_info = {
		'params_list' : true_params,
		'params_dict' : true_params_dict
	}
	if exp_data: 
		# so as not to plot "true" params for exp data
		true_params_info['params_dict'] = None

	true_params_info['true_op'] = true_op

	if pickle_file is not None:
		import pickle
		pickle.dump(
			true_params_info, 
			open(pickle_file, 'wb')
		)

def create_prior(
	true_op, 
	pickle_file=None,
	random_vals=False, 
	sigma=0.3,
	rand_min=-3, 
	rand_max=3,
	exp_data=0
):
	terms = DataBase.get_constituent_names_from_name(true_op)
	specific_terms = {}
	
	if random_vals is True:
		for term in terms:
			val = random.uniform(rand_min, rand_max)
			specific_terms[term] = [val, sigma]
	else:
		for term in terms:
			try:
				specific_terms[term] = set_prior_specific_terms[term]
			except: 
				# in case term not in set_prior_specific_terms
				val = random.uniform(rand_min, rand_max)
				specific_terms[term] = [val, sigma]
			
	if pickle_file is not None:
		import pickle
		pickle.dump(
			specific_terms, 
			open(pickle_file, 'wb')
		)
			
### Parse arguments from bash
parser = argparse.ArgumentParser(description='Pass files to pickel QHL parameters.')

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

arguments = parser.parse_args()
random_true_params = bool(arguments.random_true_params)
random_prior = bool(arguments.random_prior_terms)
exp_data = bool(arguments.use_experimental_data)

### Call functions to create picle files. 
if arguments.true_params_file is not None:
	create_qhl_params(
		true_op = arguments.true_op, 
		pickle_file=arguments.true_params_file,
		random_vals=random_true_params, 
		exp_data=exp_data
	)

if arguments.prior_file is not None:
	create_prior(
		true_op = arguments.true_op, 
		pickle_file = arguments.prior_file,
		random_vals = random_prior, 
		exp_data=exp_data
	)
