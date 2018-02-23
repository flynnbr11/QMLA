from rq import Connection, Queue
from redis import Redis
from qmd_funcs_for_rq import *

redis_conn = Redis()
q = Queue(connection=redis_conn)



### Set up command line arguments to alter script parameters. ###

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
  default=200
)

parser.add_argument(
  '-p', '--num_particles', 
  help='Number of particles to use for the learning process',
  type=int,
  default=300
)

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
parser.add_argument(
  '-pt', '--plots',
  help='True: do generate all plots for this script; False: do not.',
  type=int,
  default=0
)
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
parser.add_argument(
  '-vary_rt', '--vary_resample_threshold',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=int,
  default=0
)
parser.add_argument(
  '-vary_ra', '--vary_resample_a',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=int,
  default=0
)
parser.add_argument(
  '-vary_pgh', '--vary_pgh_factor',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=int,
  default=0
)

parser.add_argument(
  '-sq', '--store_qmd_classes',
  help='Pickle QMD classes after learning.',
  type=int,
  default=0
)
parser.add_argument(
  '-map',
  help='True to use multiprocessing.map; False to run loop.',
  type=int,
  default=0
)


arguments = parser.parse_args()
do_iqle = bool(arguments.iqle)
do_qle = bool(arguments.qle)
num_tests = arguments.num_tests
num_qubits = arguments.num_qubits
num_parameters = arguments.num_parameters
num_exp = arguments.num_experiments
num_part = arguments.num_particles
all_plots = bool(arguments.plots)
best_resample_threshold = arguments.resample_threshold
best_resample_a = arguments.resample_a
best_pgh = arguments.pgh_factor
vary_resample_a = bool(arguments.vary_resample_threshold)
vary_resample_thresh = bool(arguments.vary_resample_a)
vary_pgh_factor = bool(arguments.vary_pgh_factor)
pickle_qmd_classes = bool(arguments.store_qmd_classes)
do_map = bool(arguments.map)
#######

plot_time = get_directory_name_by_time(just_date=False) # rather than calling at separate times and causing confusion

intermediate_plots = all_plots
do_summary_plots = all_plots
store_data = all_plots

global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.

while global_true_op == 'i':
  global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.


global paulis_list
paulis_list = {'i' : np.eye(2), 'x' : evo.sigmax(), 'y' : evo.sigmay(), 'z' : evo.sigmaz()}

warnings.filterwarnings("ignore", message='Negative weights occured', category=RuntimeWarning)


#####################################

### Plotting functions 

#######################################
true_param_list=[]
true_op_list=[]

initial_op_list = ['x', 'y', 'z']
true_params = [np.random.rand() for i in range(num_parameters)]
true_param_list.append(true_params[0])
# true_op = global_true_op
true_op = random.choice(initial_op_list)
# true_op = 'xTy'
global_true_op = true_op
true_op_list.append(true_op)


pickled_qmd_directory = 'PickledQMDClasses/'+str(get_directory_name_by_time())
if not os.path.exists(pickled_qmd_directory):
  os.makedirs(pickled_qmd_directory)
                
qle_values = [] # qle True does QLE; False does IQLE
if do_qle is True:
    qle_values.append(True)
if do_iqle is True:
  qle_values.append(False)


if __name__=='__main__':
  start = time.time()
  qmd_time=0

  if do_map:
    a = time.time() # track time in just QMD
    i_range = range(num_tests)

    nprocs = cpu_count()
    pool=Pool(processes=nprocs)

    pool.map(create_and_run_qmd, i_range)
    b = time.time()
    qmd_time += b-a

  else:
    print("Using RQ")
    for i in range(num_tests):
        print("Test ", i)

        for qle in qle_values:
            a = time.time() # track time in just QMD
            
            q.enqueue(create_and_run_qmd, qmd_id=i)
#            create_and_run_qmd(qmd_id=i)

            b = time.time()
            qmd_time += b-a
  num_qle_types = do_iqle + do_qle
  num_exponentiations = num_qle_types * num_part * num_exp*num_tests

  end=time.time()
  print("\n\n\n")
  if all_plots: print("Plots made with matplotlib.")
  if not all_plots: print("Plots NOT drawn.")                                       
  print(num_qubits, "Qubits")
  print(num_qle_types, "Types of (I)QLE")
  print(num_part, "Particles")
  print(num_exp, "Experiments")
  print(num_tests, "Tests")
  print("Totalling ", num_exponentiations, " calls to ", num_qubits,"-qubit exponentiation function.")
  print("QMD-time / num exponentiations = ", qmd_time/num_exponentiations)
  print("Total time on QMD : ", qmd_time, "seconds.")
  if all_plots: print("Total time on plotting : ", d-c, "seconds.")
  print("Total time : ", end - start, "seconds.")








