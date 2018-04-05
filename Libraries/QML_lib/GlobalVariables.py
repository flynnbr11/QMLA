import argparse
default_host_name = 'localhost'
default_port_number = 6379
default_use_rq = 1
default_do_iqle  = 0 
defaul_do_qle = 1
default_use_rq = 1
default_num_runs = 1
default_num_tests = 1
default_num_qubits = 2
default_num_parameters = 2
default_num_experiments = 10
default_num_particles = 20
default_bayes_times = 5
default_do_plots =  0
default_resample_threshold = 0.5
default_resample_a = 0.95
default_pgh_factor = 1.0



class GlobalVariablesClass():
    def __init__(
        self, 
        host_name = default_host_name,
        port_number = default_port_number,
        use_rq = default_use_rq,
        do_iqle = default_do_iqle,
        do_qle = defaul_do_qle,
        num_runs = default_num_runs,
        num_tests = default_num_tests,
        num_qubits = default_num_qubits,
        num_parameters = default_num_parameters,
        num_experiments = default_num_experiments,
        num_particles = default_num_particles,
        num_times_bayes = default_bayes_times,
        all_plots = default_do_plots,
        resample_threshold = default_resample_threshold,
        resample_a = default_resample_a,
        pgh_factor = default_pgh_factor
    ):
        self.do_iqle = do_iqle
        self.do_qle = do_qle
        self.use_rq = use_rq
        self.num_runs = num_runs
        self.num_tests = num_tests
        self.num_qubits = num_qubits
        self.num_parameters = num_parameters
        self.num_experiments = num_experiments
        self.num_particles = num_particles
        self.num_times_bayes = num_times_bayes
        self.all_plots = all_plots
        self.resample_threshold = resample_threshold
        self.resample_a = resample_a
        self.pgh_factor = pgh_factor



def parse_cmd_line_args(args):

    parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

    # Add parser arguments, ie command line arguments for QMD
    ## QMD parameters -- fundamentals such as number of particles etc
    parser.add_argument(
      '-r', '--num_runs', 
      help="Number of runs to perform majority voting.",
      type=int,
      default=1
    )
    parser.add_argument(
      '-t', '--num_tests', 
      help="Number of complete tests to average over.",
      type=int,
      default=1
    )

    parser.add_argument(
      '-e', '--num_experiments', 
      help='Number of experiments to use for the learning process',
      type=int,
      default=5
    )
    parser.add_argument(
      '-p', '--num_particles', 
      help='Number of particles to use for the learning process',
      type=int,
      default=10
    )
    parser.add_argument(
      '-bt', '--bayes_times', 
      help='Number of times to consider in Bayes function.',
      type=int,
      default=2
    )
    parser.add_argument(
      '-rq', '--use_rq', 
      help='Bool whether to use RQ for parallel or not.',
      type=int,
      default=1
    )

    ## Parameters about the model to use as true model (currently deprecated)
    parser.add_argument(
      '-q', '--num_qubits', 
      help='Number of qubits to run tests for.',
      type=int,
      default=2
    )
    parser.add_argument(
      '-pm', '--num_parameters', 
      help='Number of parameters to run tests for.',
      type=int,
      default=1
    )

    ## Whether to use QLE, IQLE or both (currently deprecated)
    parser.add_argument(
      '-qle',
      help='True to perform QLE, False otherwise.',
      type=int,
      default=1
    )
    parser.add_argument(
      '-iqle',
      help='True to perform IQLE, False otherwise.',
      type=int,
      default=1
    )

    ## Include optional plots
    parser.add_argument(
      '-pt', '--plots',
      help='True: do generate all plots for this script; False: do not.',
      type=int,
      default=0
    )

    ## QInfer parameters, i.e. resampling a and resamping threshold, pgh prefactor.
    parser.add_argument(
      '-rt', '--resample_threshold',
      help='Resampling threshold for QInfer.',
      type=float,
      default=0.6
    )
    parser.add_argument(
      '-ra', '--resample_a',
      help='Resampling a for QInfer.',
      type=float,
      default=0.9
    )
    parser.add_argument(
      '-pgh', '--pgh_factor',
      help='Resampling threshold for QInfer.',
      type=float,
      default=1.0
    )

    # Process arguments from command line
    arguments = parser.parse_args(args)
    
    do_iqle = bool(arguments.iqle)
    do_qle = bool(arguments.qle)
    use_rq = bool(arguments.use_rq)
    num_runs = arguments.num_runs
    num_tests = arguments.num_tests
    num_qubits = arguments.num_qubits
    num_parameters = arguments.num_parameters
    num_experiments = arguments.num_experiments
    num_particles = arguments.num_particles
    num_times_bayes = arguments.bayes_times
    if num_times_bayes > num_experiments:
        num_times_bayes = num_experiments-1
    all_plots = bool(arguments.plots)
    resample_threshold = arguments.resample_threshold
    resample_a = arguments.resample_a
    pgh_factor = arguments.pgh_factor
    
    
    # Use arguments to initialise global variables class. 
#    global global_variables
    global_variables = GlobalVariablesClass(
        do_iqle = do_iqle,
        do_qle = do_qle,
        use_rq = use_rq,
        num_runs = num_runs,
        num_tests = num_tests,
        num_qubits = num_qubits,
        num_parameters = num_parameters,
        num_experiments = num_experiments,
        num_particles = num_particles,
        num_times_bayes = num_times_bayes,
        all_plots = arguments.plots,
        resample_threshold = resample_threshold,
        resample_a = resample_a,
        pgh_factor = pgh_factor
    )
    
    return global_variables

