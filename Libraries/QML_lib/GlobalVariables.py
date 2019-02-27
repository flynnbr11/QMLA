import argparse
import os
import sys
import pickle 


import UserFunctions

"""
This file is callable with *kwargs from a separate QMD program. 
It returns an instance of the class GlobalVariablesClass, which has attributes 
for all the user defined parameters, and defaults if not specified by the user. 

"""

def get_directory_name_by_time(just_date=False):
    import datetime
    # Directory name based on date and time it was generated 
    # from https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    now =  datetime.date.today()
    year = now.strftime("%y")
    month = now.strftime("%b")
    day = now.strftime("%d")
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    date = str (str(day)+'_'+str(month)+'_'+str(year) )
    time = str(str(hour)+'_'+str(minute))
    name = str(date+'/'+time+'/')
    if just_date is False:
        return name
    else: 
        return str(date+'/')

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time


def log_print(to_print_list, log_file):
    identifier = str(str(time_seconds()) +" [GLOBAL VARIABLES]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, 
            str(to_print),
            file=write_log_file,
            flush=True
        )


default_host_name = 'localhost'
default_port_number = 6379
default_use_rq = 1
default_do_iqle  = 0 
default_do_qle = 1
default_use_rq = 1
default_num_runs = 1
default_num_tests = 1
default_num_qubits = 2
default_num_parameters = 2
default_num_experiments = 10
default_num_particles = 20
default_bayes_times = 5
default_bayes_threshold_lower = 1
default_bayes_threshold_upper = 100
default_gaussian = True
default_custom_prior = False
default_do_plots =  0
default_resample_threshold = 0.5
default_resample_a = 0.95
default_pgh_factor = 1.0
default_qmd_id = 1
default_results_directory = get_directory_name_by_time(
  just_date=False
)
default_pickle_qmd_class = 0
default_port_number = 6379
default_host = 'localhost'
default_rq_timeout = 3600
default_log_file = 'default_log_file.log'
default_save_plots = False
default_cumulative_csv = 'cumulative_csv.csv'
default_measurement_type = 'full_access'
default_experimental_data = False
# NOTE true operator is set in dict in UserFunctions: default_true_operators_by_generator
default_true_operator = 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
default_qhl_test = 0
default_further_qhl = 0
default_dataset = 'NV_HahnPeaks_expdataset'
default_data_max_useful_time = 2000 # nanoseconds
default_data_time_offset = 180 # nanoseconds
default_growth_generation_rule = 'two_qubit_ising_rotation_hyperfine'
default_prior_pickle_file = None
default_true_params_pickle_file = None
default_true_expec_path = None
default_latex_mapping_file = str(
  default_results_directory +
  '/LatexMapping.txt'
)
default_plot_probe_file = None
default_reallocate_resources=0
default_bayes_time_binning=0



class GlobalVariablesClass():
    def __init__(
        self, 
        arguments, 
        **kwargs
    ):
        # self.true_operator = true_operator
        self.growth_generation_rule = arguments.growth_generation_rule
        self.alternative_growth_rules = arguments.alternative_growth_rules
        self.multiQHL = bool(arguments.multiQHL)
        self.models_for_qhl = arguments.models_for_qhl
        self.prior_pickle_file = arguments.prior_pickle_file
        self.true_params_pickle_file = arguments.true_params_pickle_file
        true_params_info = pickle.load(
            open(self.true_params_pickle_file, 'rb')
        )
        self.true_operator = true_params_info['true_op']
        self.true_params = true_params_info['params_list']
        # self.true_operator = UserFunctions.default_true_operators_by_generator[
        #     self.growth_generation_rule
        # ]
        self.qhl_test = bool(arguments.qhl_test)
        self.further_qhl = bool(arguments.further_qhl)
        self.do_iqle = bool(arguments.do_iqle)
        self.do_qle = bool(arguments.do_qle)
        self.use_rq = bool(arguments.use_rq)
        self.num_runs = arguments.num_runs
        self.num_tests = arguments.num_tests
        self.num_qubits = arguments.num_qubits
        self.num_parameters = arguments.num_parameters
        self.num_experiments = arguments.num_experiments
        self.num_particles = arguments.num_particles
        self.num_times_bayes = arguments.num_times_bayes
        self.bayes_lower = arguments.bayes_lower
        self.bayes_upper = arguments.bayes_upper
        self.save_plots = bool(arguments.save_plots)
        self.gaussian = bool(arguments.gaussian)
        self.custom_prior = bool(arguments.custom_prior)
        self.resample_threshold = arguments.resample_threshold
        self.resample_a = arguments.resample_a
        self.pgh_factor = arguments.pgh_factor
        self.pickle_qmd_class = bool(arguments.pickle_qmd_class)
        self.qmd_id = arguments.qmd_id
        self.host_name = arguments.host_name
        self.port_number = arguments.port_number
#        self.results_directory = 'Results/'+results_directory
        self.results_directory = arguments.results_directory

        self.rq_timeout = arguments.rq_timeout
        self.log_file = arguments.log_file
        # self.save_plots = bool(arguments.save_plots)
        self.cumulative_csv = arguments.cumulative_csv
        self.use_experimental_data = bool(arguments.experimental_data)
        # self.measurement_type = arguments.measurement_type
        self.measurement_type = UserFunctions.get_measurement_type(
          growth_generator = self.growth_generation_rule
        )
        # self.dataset = arguments.dataset
        self.dataset = UserFunctions.get_experimental_dataset(
          growth_generator = self.growth_generation_rule
        )
        self.data_time_offset = arguments.data_time_offset
        self.data_max_time = arguments.data_max_time 
        self.true_expec_path = arguments.true_expec_path
        self.plot_probe_file = arguments.plot_probe_file
        self.special_probe = arguments.special_probe_for_learning
        self.latex_mapping_file = arguments.latex_mapping_file
        self.reallocate_resources = arguments.reallocate_resources
        self.param_min = arguments.param_min
        self.param_max = arguments.param_max
        self.param_mean = arguments.param_mean
        self.param_sigma = arguments.param_sigma
        self.bayes_time_binning = bool(arguments.bayes_time_binning)
        self.bayes_factors_use_all_exp_times = bool(arguments.bayes_factors_use_all_exp_times)
        self.num_probes = arguments.num_probes
        self.probe_noise_level = arguments.probe_noise_level

        if self.results_directory[-1] != '/':
            self.results_directory += '/'
        self.plots_directory = self.results_directory+'plots/'


        if not os.path.exists(self.results_directory):
            try:
                os.makedirs(self.results_directory)
            except FileExistsError:
                pass
                        

        if not os.path.exists(self.plots_directory):
            try:
                os.makedirs(self.plots_directory)
            except FileExistsError:
                pass

        self.long_id ='{0:03d}'.format(self.qmd_id)
        if self.further_qhl==True:
          self.results_file = self.results_directory+'further_qhl_results_'+str(self.long_id)+'.p' #for pickling results into
          self.class_pickle_file = self.results_directory+'further_qhl_qmd_class_'+str(self.long_id)+'.p'
        else:
          self.results_file = self.results_directory+'results_'+str(self.long_id)+'.p' #for pickling results into
          self.class_pickle_file = self.results_directory+'qmd_class_'+str(self.long_id)+'.p'
        


def parse_cmd_line_args(args):

    parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

    # Add parser arguments, ie command line arguments for QMD

    # parser.add_argument(
    #   '-op', '--true_operator', 
    #   help="True operator to be simulated and learned against.",
    #   type=str,
    #   default=default_true_operator
    # )

    parser.add_argument(
      '-qhl', '--qhl_test', 
      help="Bool to test QHL on given true operator only.",
      type=int,
      default=default_qhl_test
    )

    parser.add_argument(
      '-fq', '--further_qhl', 
      help="Bool to perform further QHL on best models from previous run.",
      type=int,
      default=default_further_qhl
    )


    ## QMD parameters -- fundamentals such as number of particles etc
    parser.add_argument(
      '-r', '--num_runs', 
      help="Number of runs to perform majority voting.",
      type=int,
      default=default_num_runs
    )

    parser.add_argument(
      '-t', '--num_tests', 
      help="Number of complete tests to average over.",
      type=int,
      default=default_num_tests
    )

    parser.add_argument(
      '-e', '--num_experiments', 
      help='Number of experiments to use for the learning process',
      type=int,
      default=default_num_experiments
    )
    parser.add_argument(
      '-p', '--num_particles', 
      help='Number of particles to use for the learning process',
      type=int,
      default=default_num_particles
    )
    parser.add_argument(
      '-bt', '--num_times_bayes', 
      help='Number of times to consider in Bayes function.',
      type=int,
      default=default_bayes_times
    )
    parser.add_argument(
      '-rq', '--use_rq', 
      help='Bool whether to use RQ for parallel or not.',
      type=int,
      default=default_use_rq
    )

    parser.add_argument(
      '-bu', '--bayes_upper', 
      help='Higher Bayes threshold.',
      type=int,
      default=default_bayes_threshold_upper
    )

    parser.add_argument(
      '-bl', '--bayes_lower', 
      help='Lower Bayes threshold.',
      type=int,
      default=default_bayes_threshold_lower
    )

    ## Parameters about the model to use as true model (currently deprecated)
    parser.add_argument(
      '-q', '--num_qubits', 
      help='Number of qubits to run tests for.',
      type=int,
      default=default_num_qubits
    )
    parser.add_argument(
      '-pm', '--num_parameters', 
      help='Number of parameters to run tests for.',
      type=int,
      default=default_num_parameters
    )

    ## Whether to use QLE, IQLE or both (currently deprecated)
    parser.add_argument(
      '-qle', '--do_qle',
      help='True to perform QLE, False otherwise.',
      type=int,
      default=default_do_qle
    )
    parser.add_argument(
      '-iqle', '--do_iqle',
      help='True to perform IQLE, False otherwise.',
      type=int,
      default=default_do_iqle
    )

    parser.add_argument(
      '-g', '--gaussian',
      help='True: normal distribution; False: uniform.',
      type=int,
      default=default_gaussian
    )

    parser.add_argument(
      '-cpr', '--custom_prior',
      help='True: use custom prior given to QMD instance; False: use defulat.',
      type=int,
      default=default_custom_prior
    )
    

    ## Include optional plots
    parser.add_argument(
      '-pt', '--save_plots',
      help='True: save all plots for this QMD; False: do not.',
      type=int,
      default=default_save_plots
    )

    ## QInfer parameters, i.e. resampling a and resamping threshold, pgh prefactor.
    parser.add_argument(
      '-rt', '--resample_threshold',
      help='Resampling threshold for QInfer.',
      type=float,
      default=default_resample_threshold
    )
    parser.add_argument(
      '-ra', '--resample_a',
      help='Resampling a for QInfer.',
      type=float,
      default=default_resample_a
    )
    parser.add_argument(
      '-pgh', '--pgh_factor',
      help='Resampling threshold for QInfer.',
      type=float,
      default=default_pgh_factor
    )


    ## Redis environment
    parser.add_argument(
      '-host', '--host_name',
      help='Name of Redis host.',
      type=str,
      default=default_host
    )
    parser.add_argument(
      '-port', '--port_number',
      help='Redis port number.',
      type=int,
      default=default_port_number
    )


    parser.add_argument(
      '-qid', '--qmd_id',
      help='ID tag for QMD.',
      type=int,
      default=default_qmd_id
    )
    parser.add_argument(
      '-dir', '--results_directory',
      help='Relative directory to store results in.',
      type=str,
      default=default_results_directory
    )
    parser.add_argument(
      '-pkl', '--pickle_qmd_class',
      help='Store QMD class in pickled file at end. Large memory requirement, recommend not to.',
      type=int,
      default=default_pickle_qmd_class
    )

    parser.add_argument(
      '-rqt', '--rq_timeout',
      help='Time allowed before RQ job crashes.',
      type=int,
      default=default_rq_timeout
    )
    
    parser.add_argument(
      '-log', '--log_file',
      help='File to log RQ workers.',
      type=str,
      default=default_log_file
    )
    
    parser.add_argument(
      '-cb', '--cumulative_csv',
      help='CSV to store Bayes factors of all QMDs.',
      type=str,
      default=default_cumulative_csv
    )
    
    parser.add_argument(
      '-exp', '--experimental_data',
      help='Use experimental data if provided',
      type=int,
      default=default_experimental_data
    )
    # parser.add_argument(
    #   '-meas', '--measurement_type',
    #   help='Which measurement type to use. Must be written in Evo.py.',
    #   type=str,
    #   default=default_measurement_type
    # )

    # parser.add_argument(
    #   '-ds', '--dataset',
    #   help='Dataset to use',
    #   type=str,
    #   default=default_dataset
    # )

    parser.add_argument(
      '-dst', '--data_max_time',
      help='Maximum useful time in given data.',
      type=int,
      default=default_data_max_useful_time
    )

    parser.add_argument(
      '-dto', '--data_time_offset',
      help='Offset to ensure at t=0, Pr=1.',
      type=int,
      default=default_data_time_offset
    )

    parser.add_argument(
      '-bintimes', '--bayes_time_binning',
      help='Store QMD class in pickled file at end. Large memory requirement, recommend not to.',
      type=int,
      default=default_bayes_time_binning
    )

    parser.add_argument(
      '-bftimesall', '--bayes_factors_use_all_exp_times',
      help='Store QMD class in pickled file at end. Large memory requirement, recommend not to.',
      type=int,
      default=0
    )

    parser.add_argument(
      '-ggr', '--growth_generation_rule',
      help='Rule applied for generation of new models during QMD. \
        Corresponding functions must be built into ModelGeneration',
      type=str,
      default=default_growth_generation_rule
    )

    parser.add_argument(
      '-agr', '--alternative_growth_rules',
      help='Growth rules to form other trees.',
      # type=str,
      action='append',
      default=[],
    )

    parser.add_argument(
      '-qhl_mods', '--models_for_qhl',
      help='Models on which to run QHL.',
      # type=str,
      action='append',
      default=[],
    )
    
    parser.add_argument(
      '-mqhl', '--multiQHL',
      help='Run QHL test on multiple (provided) models.',
      type=int,
      default=0
    )

    parser.add_argument(
      '-prior_path', '--prior_pickle_file',
      help='Path to save prior to.',
      type=str,
      default=default_prior_pickle_file
    )
    parser.add_argument(
      '-true_params_path', '--true_params_pickle_file',
      help='Path to save true params to.',
      type=str,
      default=default_true_params_pickle_file
    )

    parser.add_argument(
      '-true_expec_path', '--true_expec_path',
      help='Path to save true params to.',
      type=str,
      default=default_true_params_pickle_file
    )
    parser.add_argument(
      '-plot_probes', '--plot_probe_file',
      help='Path where plot probe dict is pickled to.',
      type=str,
      default=default_plot_probe_file
    )

    parser.add_argument(
      '-special_probe', '--special_probe_for_learning',
      help='Specify type of probe to use during learning.',
      type=str,
      default=None
    )

    parser.add_argument(
      '-latex', '--latex_mapping_file',
      help='Path to save list of terms latex/name maps to.',
      type=str,
      default=default_latex_mapping_file
    )

    parser.add_argument(
      '-resource', '--reallocate_resources',
      help='Bool: whether to reallocate resources scaling  \
        with num qubits/terms to be learned during QHL.',
      type=int,
      default=default_reallocate_resources
    )

    parser.add_argument(
      '-pmin', '--param_min',
      help='Minimum valid paramater value.',
      type=float,
      default=0
    )
    parser.add_argument(
      '-pmax', '--param_max',
      help='Maximum valid paramater value.',
      type=float,
      default=1
    )

    parser.add_argument(
      '-pmean', '--param_mean',
      help='Default mean parameter value for normal distribution.',
      type=float,
      default=0.5
    )

    parser.add_argument(
      '-psigma', '--param_sigma',
      help='Default std dev on distribution',
      type=float,
      default=0.5
    )

    parser.add_argument(
      '-nprobes', '--num_probes',
      help='How many probe states in rota for learning parameters.',
      type=int,
      default=20
    )

    parser.add_argument(
      '-pnoise', '--probe_noise_level',
      help='Noise level to add to probe for learning',
      type=float,
      default=0.03
    )

    # Process arguments from command line
    arguments = parser.parse_args(args)
    
    # Use arguments to initialise global variables class. 
    global_variables = GlobalVariablesClass(
        arguments,
    )

    # args_dict = vars(arguments)
    args_dict = vars(global_variables)

    for a in list(args_dict.keys()):
      log_print(
        [
        a, 
        ':', 
        args_dict[a]
        ],
        log_file = global_variables.log_file
      )

    return global_variables



