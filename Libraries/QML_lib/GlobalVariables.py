import argparse
import os, sys
import pickle 

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
default_results_directory = get_directory_name_by_time(just_date=False)
default_pickle_qmd_class = 0
default_port_number = 6379
default_host = 'localhost'
default_rq_timeout = 3600
default_log_file = 'default_log_file.log'
default_save_plots = False
default_cumulative_csv = 'cumulative_bayes.csv'
default_experimental_data = False
default_true_operator = 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
default_qhl_test = 0
default_dataset = 'NV_HahnPeaks_expdataset'
default_data_max_useful_time = 2000 # nanoseconds
default_data_time_offset = 180 # nanoseconds

class GlobalVariablesClass():
    def __init__(
        self, 
        true_operator = default_true_operator,
        qhl_test = default_qhl_test,
        use_rq = default_use_rq,
        do_iqle = default_do_iqle,
        do_qle = default_do_qle,
        num_runs = default_num_runs,
        num_tests = default_num_tests,
        num_qubits = default_num_qubits,
        num_parameters = default_num_parameters,
        num_experiments = default_num_experiments,
        num_particles = default_num_particles,
        num_times_bayes = default_bayes_times,
        bayes_upper = default_bayes_threshold_upper,
        bayes_lower = default_bayes_threshold_lower,
        all_plots = default_do_plots,
        gaussian = default_gaussian,
        custom_prior = default_custom_prior,
        resample_threshold = default_resample_threshold,
        resample_a = default_resample_a,
        pgh_factor = default_pgh_factor,
        qmd_id = default_qmd_id,
        host_name = default_host,
        port_number = default_port_number,
        results_directory = default_results_directory,
        pickle_qmd_class = default_pickle_qmd_class,
        rq_timeout = default_rq_timeout,
        log_file = default_log_file,
        save_plots = default_save_plots,
        cumulative_csv = default_cumulative_csv,
        experimental_data = default_experimental_data,
        dataset = default_dataset,
        data_max_time = default_data_max_useful_time,
        data_time_offset = default_data_time_offset

    ):
        self.true_operator = true_operator
        self.qhl_test = qhl_test
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
        self.bayes_lower = bayes_lower
        self.bayes_upper = bayes_upper
        self.all_plots = all_plots
        self.gaussian = gaussian
        self.custom_prior = custom_prior
        self.resample_threshold = resample_threshold
        self.resample_a = resample_a
        self.pgh_factor = pgh_factor
        self.pickle_qmd_class = pickle_qmd_class
        self.qmd_id = qmd_id
        self.host_name = host_name
        self.port_number = port_number
#        self.results_directory = 'Results/'+results_directory
        self.results_directory = results_directory

        self.rq_timeout = rq_timeout
        self.log_file = log_file
        self.save_plots = save_plots
        self.cumulative_csv = cumulative_csv
        self.use_experimental_data = experimental_data
        self.dataset = dataset
        self.data_time_offset = data_time_offset
        self.data_max_time = data_max_time 


        if self.results_directory[-1] != '/':
            self.results_directory += '/'
        self.plots_directory = self.results_directory+'Plots/'


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
        self.results_file = self.results_directory+'results_'+str(self.long_id)+'.p' #for pickling results into
        self.class_pickle_file = self.results_directory+'qmd_class_'+str(self.long_id)+'.p'
        


def parse_cmd_line_args(args):

    parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

    # Add parser arguments, ie command line arguments for QMD

    parser.add_argument(
      '-op', '--true_operator', 
      help="True operator to be simulated and learned against.",
      type=str,
      default=default_true_operator
    )

    parser.add_argument(
      '-qhl', '--qhl_test', 
      help="Bool to test QHL on given true operator only.",
      type=int,
      default=default_qhl_test
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
      '-bt', '--bayes_times', 
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
      '-qle',
      help='True to perform QLE, False otherwise.',
      type=int,
      default=default_do_qle
    )
    parser.add_argument(
      '-iqle',
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
      '-pt', '--plots',
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
      '-host', '--redis_host',
      help='Name of Redis host.',
      type=str,
      default=default_host
    )
    parser.add_argument(
      '-port', '--redis_port_number',
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
      '-pkl', '--pickle_result_class',
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
      '-log', '--logfile',
      help='File to log RQ workers.',
      type=str,
      default=default_log_file
    )
    
    parser.add_argument(
      '-cb', '--cumulative_bayes',
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

    parser.add_argument(
      '-ds', '--dataset',
      help='Dataset to use',
      type=str,
      default=default_dataset
    )

    parser.add_argument(
      '-dst', '--dataset_max_time',
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

    # Process arguments from command line
    arguments = parser.parse_args(args)
    
    true_operator = arguments.true_operator
    qhl_test = bool(arguments.qhl_test)
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
    bayes_upper = arguments.bayes_upper
    bayes_lower = arguments.bayes_lower
    all_plots = bool(arguments.plots)
    gaussian = bool(arguments.gaussian)
    custom_prior = bool(arguments.custom_prior)
    resample_threshold = arguments.resample_threshold
    resample_a = arguments.resample_a
    pgh_factor = arguments.pgh_factor
    qmd_id = arguments.qmd_id
    host_name = arguments.redis_host
    port_number = arguments.redis_port_number
    results_directory = arguments.results_directory
    pickle_qmd_class = bool(arguments.pickle_result_class)
    rq_timeout = arguments.rq_timeout
    log_file = arguments.logfile
    cumulative_csv = arguments.cumulative_bayes
    use_experimental_data = bool(arguments.experimental_data)
    dataset = arguments.dataset
    data_max_time = arguments.dataset_max_time    
    data_time_offset = arguments.data_time_offset
    
    # Use arguments to initialise global variables class. 
    global_variables = GlobalVariablesClass(
        true_operator = true_operator,
        qhl_test = qhl_test, 
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
        bayes_lower = bayes_lower, 
        bayes_upper = bayes_upper, 
        resample_threshold = resample_threshold,
        resample_a = resample_a,
        pgh_factor = pgh_factor,
        gaussian = gaussian,
        custom_prior = custom_prior,
        qmd_id = qmd_id, 
        host_name = host_name,
        port_number = port_number,
        results_directory = results_directory,
        pickle_qmd_class = pickle_qmd_class,
        rq_timeout = rq_timeout,
        log_file = log_file,
        save_plots = all_plots,
        cumulative_csv = cumulative_csv,
        experimental_data = use_experimental_data,
        dataset = dataset,
        data_max_time = data_max_time,
        data_time_offset = data_time_offset
    )

    return global_variables



