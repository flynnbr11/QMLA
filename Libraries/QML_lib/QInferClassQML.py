from __future__ import print_function # so print doesn't show brackets

import qinfer as qi
import numpy as np
import scipy as sp
import warnings

# from Evo import * # TODO remove ALL import * calls across QMD
import Evo
import ExperimentalDataFunctions as expdt
import GrowthRules
# from ProbeStates import *
from MemoryTest import print_loc, print_file_line
import ProbeGeneration
from psutil import virtual_memory
import DataBase

global_print_loc=False
global debug_print
debug_print = False
global debug_log_print
debug_log_print = False
global likelihood_dev
likelihood_dev = False
global debug_print_file_line
debug_print_file_line=False

def time_seconds():
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
        log_identifier='[GenSim]'
    identifier = str(
        str(time_seconds()) 
        + " [QML-Qinfer] ("
        + str(log_identifier) 
        + ")]"
    )
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file)


class QInferModelQML(qi.FiniteOutcomeModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :param np.array : :math:`\left|+\Psi\rangle` state under 
    a Hamiltonian :math:`H = \omega \sum_i \gamma_i / 2`
    of a set of (Pauli) operators given by 
    :param np.array oplist:  
    using the interactive QLE model proposed by [WGFC13a]_.
    
    :param np.array oplist: Set of operators whose sum 
    defines the evolution Hamiltonian

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of
        degenerate models; there is no "good" reason to ever set this other
        than zero, other than to test with an explicitly broken model.
        
    :param str solver: Which solver to use for the Hamiltonian simulation.
        'scipy' invokes matrix exponentiation (i.e. time-independent evolution)
        -> fast, accurate when applicable
        'qutip' invokes ODE solver (i.e. time-dependent evolution can 
        be also managed approx.)
        -> not invoked by deafult
    """
    
    ## INITIALIZER ##

    def __init__(
        self, 
        oplist, 
        modelparams, # modelparams not needed (used) for this class # TODO remove unneeded inputs
        probecounter=None, 
        use_time_dep_true_model = False,
        time_dep_true_params = None,
        num_time_dep_true_params = 0,
        true_oplist = None, 
        truename=None, 
        num_probes=40, 
        probe_dict=None,
        sim_probe_dict=None, 
        trueparams=None, 
        probelist=None, 
        min_freq=0, 
        solver='scipy', 
        measurement_type = 'full_access',
        growth_generation_rule = None, 
        use_experimental_data=False, 
        experimental_measurements = None,
        experimental_measurement_times=None,
        trotter=False, 
        qle=True,
        use_exp_custom=True, 
        exp_comparison_tol=None,
        enable_sparse=True, 
        model_name=None,
        log_file='QMDLog.log',
        log_identifier=None
    ):
        self._solver = solver 
        # This is the solver used for time evolution scipy is faster
        # QuTip can handle implicit time dependent likelihoods
        self._oplist = oplist
        self._probecounter = probecounter
        self._a = 0
        self._b = 0 
        self.QLE = qle
        self._trotter = trotter
        self._modelparams = modelparams
        # print("[QInferModelQML] \n Oplist: {} \n params: {}".format(
        #     self._oplist, 
        #     self._modelparams
        #     )
        # )
        self.signs_of_inital_params = np.sign(modelparams)
        self._true_oplist = true_oplist
        self._trueparams = trueparams
        self._truename = truename
        # print("[QML Qinfer class] True op list:", self._true_oplist)
        self._true_dim = DataBase.get_num_qubits(self._truename)
        self.use_time_dep_true_model = use_time_dep_true_model
        self.time_dep_true_params = time_dep_true_params
        self.num_time_dep_true_params = num_time_dep_true_params
        self.measurement_type = measurement_type
        self.use_experimental_data = use_experimental_data
        self.log_file = log_file
        self.growth_generation_rule = growth_generation_rule
        try:
            self.growth_class = GrowthRules.get_growth_generator_class(
                growth_generation_rule = self.growth_generation_rule,
                use_experimental_data = self.use_experimental_data,
                log_file = self.log_file
            )
        except:
            self.growth_class = None

        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_times = experimental_measurement_times
        self.use_exp_custom = use_exp_custom
        self.enable_sparse = enable_sparse
        self.exp_comparison_tol = exp_comparison_tol  
        self._min_freq = min_freq
        self.ModelName = model_name
        self.model_dimension = DataBase.get_num_qubits(self.ModelName)
        self.inBayesUpdates = False
        self.ideal_probe = None
        # self.IdealProbe = DataBase.ideal_probe(self.ModelName)
        self.ideal_probelist = None
        self.log_identifier = log_identifier
        if true_oplist is not None and trueparams is None:
            raise(
                ValueError(
                    '\nA system Hamiltonian with unknown \
                    parameters was requested'
                )
            )
        if true_oplist is None:
            warnings.warn(
                "\nI am assuming the Model and System \
                Hamiltonians to be the same", UserWarning
            )
            self._trueHam = None
        else:
            self._trueHam = None

        super(QInferModelQML, self).__init__(self._oplist)
        
       
        self.NumProbes = num_probes
        if probe_dict is None: 
            self.log_print(
                [
                    "Generating random probes"
                ]
            )
            self.probe_dict = ProbeGeneration.seperable_probe_dict(
                max_num_qubits=12, 
                num_probes = self.NumProbes
            ) # TODO -- make same as number of qubits in model.
            self.sim_probe_dict = self.probe_dict
        else:
            self.probe_dict = probe_dict   
            self.sim_probe_dict = sim_probe_dict

        # log_print(
        #     [
        #         "Mod name:", self.ModelName,
        #         "n_modelparams:", self.n_modelparams,
        #         "probe[(0,1)]:", 
        #         self.probe_dict[(0,1)],
        #         "\nsim probe[(0,1)]:", 
        #         self.sim_probe_dict[(0,1)],
        #     ],
        #     self.log_file, 
        #     self.log_identifier
        # )

    ## PROPERTIES ##
    @property
    def n_modelparams(self):
        return len(self._oplist)

    # Modelparams is the list of parameters in the System Hamiltonian
    # -- the ones we want to know
    # Possibly add a second axis to modelparams.    
    @property
    def modelparam_names(self):
        modnames = ['w0']
        for modpar in range(self.n_modelparams-1):
            modnames.append('w' + str(modpar+1))
        return modnames

    # expparams are the {t, w1, w2, ...} guessed parameters, i.e. each element 
    # is a particle with a specific sampled value of the corresponding parameter
  
    @property
    def expparams_dtype(self):
        expnames = [('t', 'float')]
        for exppar in range(self.n_modelparams):
            expnames.append(('w_' + str(exppar+1), 'float'))
        return expnames
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##

    def are_models_valid(self, modelparams):
        # Before setting new distribution after resampling, 
        # checks that all parameters have same sign as the 
        # initial given parameter for that term. 
        # Otherwise, redraws the distribution. 
        same_sign_as_initial = False
        if same_sign_as_initial==True:
            new_signs = np.sign(modelparams)
            validity_by_signs=np.all(
                np.sign(modelparams) == self.signs_of_inital_params,
                axis=1
            )
            return validity_by_signs
        else:
            validity = np.all(np.abs(modelparams) > self._min_freq, axis=1)
            return validity

    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
        

    def likelihood(
        self, 
        outcomes, 
        modelparams, 
        expparams
    ):
        # log_print(
        #     ["likelihood function"],
        #     self.log_file, 
        #     self.log_identifier
        # )
        super(QInferModelQML, self).likelihood(
            outcomes, modelparams, expparams
        ) # just adds to self._call_count (Qinfer abstact model class)
        print_file_line(debug_print_file_line)
        # log_print(
        #     [
        #     'Likelihood function. \t len(modelparams):',
        #     len(modelparams)
        #     ],
        #     self.log_file, 
        #     self.log_identifier
        # )
        
        import copy
        print_loc(global_print_loc)
        cutoff=min(len(modelparams), 5)
        num_particles = modelparams.shape[0]
        self._a += 1
        if self._a % 2 == 1:
            self._b += 1
        num_parameters = modelparams.shape[1]
        # true_dim = np.log2(self._true_oplist[0].shape[0])
        # sim_dim = np.log2(self._oplist[0].shape[0])
        
        if  num_particles == 1:
            print_file_line(debug_print_file_line)
            # print("[likelihood] true_evo. outcomes", outcomes)
            sample = np.array([expparams.item(0)[1:]])[0:num_parameters]
            true_evo = True
            operators = self._true_oplist
            params = [copy.deepcopy(self._trueparams)]
            
            if self.use_time_dep_true_model:
                # Multiply time dependent parameters by time of this evolution.
                time = expparams['t'] 
                a=len(params[0])-self.num_time_dep_true_params
                b=len(params[0])
                before = (params)
                for i in range(a,b):
                    # Because params is a list of 1 element, an array, need [0] index.
                    params[0][i] *=  time
            ham_num_qubits = self._true_dim
        else:
            print_file_line(debug_print_file_line)
            # print("[likelihood] sim_evo. outcomes", outcomes)
            sample = np.array([expparams.item(0)[1:]])
            true_evo = False
            operators = self._oplist
            params = modelparams
            ham_num_qubits = self.model_dimension
        # ham_num_qubits = np.log2(operators[0].shape[0])

        # Now get pr0 and pass to likelihood function
        # log_print(
        #     [
        #     '[Likelihood function]',
        #     'true evo:', true_evo, 
        #     'self._b=', self._b
        #     ],
        #     self.log_file, 
        #     self.log_identifier
        # )


        if (
            true_evo == True
            and 
            self.use_experimental_data == True
        ):
            time = expparams['t']
            # print(
            #     "[likelihood fnc] Experimental data being called.",
            #     # "\nProbe", probe
            # )
            if debug_log_print:
                log_print(
                    [
                    'Getting system outcome',
                    'time:\n', time
                    ],
                    self.log_file, 
                    self.log_identifier
                )
            #print("Time:", time[0])
            try:
                # If time already exists in experimental data
                experimental_expec_value = self.experimental_measurements[time]
            except:
                #print("t=",time,"not found in data")
                #print("t type:", type(time))
                experimental_expec_value = expdt.nearestAvailableExpVal(
                    times = self.experimental_measurement_times,
                    experimental_data = self.experimental_measurements,
                    t = time
                )
            if debug_log_print:
                log_print(
                    [
                    "Using experimental time",time,
                    "\texp val:", experimental_expec_value
                    ],
                    self.log_file, 
                    self.log_identifier
                )
            pr0 = np.array([[experimental_expec_value]])

        else:  
            print_file_line(debug_print_file_line)
            # print(
            #     "[likelihood fnc] Simulator being called.",
            #     # "\nProbe", probe
            # )

            # if self.inBayesUpdates:
            #     if self.ideal_probe is not None:
            #         probe = self.ideal_probe # this won't work
            #     elif self.ideal_probelist is not None: 
            #         probe = self.ideal_probelist[self._b % 2] # this won't work
            #     else:
            #         print(
            #             "Either ideal_probe or ideal_probes \
            #             must be given"
            #         )
            # else:
            #     probe = self.probe_dict[
            #         (self._b % int(self.NumProbes)), 
            #         ham_num_qubits
            #     ]

            if true_evo == True:
                print_file_line(debug_print_file_line)
                # print("[likelihood] trying to get probe id ", 
                #     (self._b % int(self.NumProbes)), 
                #     ham_num_qubits
                # )
                probe = self.probe_dict[
                    (self._b % int(self.NumProbes)), 
                    ham_num_qubits
                ]
                print_file_line(debug_print_file_line)
            else:
                print_file_line(debug_print_file_line)
                probe = self.sim_probe_dict[
                    (self._b % int(self.NumProbes)), 
                    ham_num_qubits
                ]
            print_file_line(debug_print_file_line)
            # print(
            #     "\n\n[likelihood fnc]",
            #     "\n True evo:", true_evo, 
            #     "\nProbe", probe, "\n\n"
            # )
            
            ham_minus = np.tensordot(
                sample, 
                self._oplist, 
                axes=1
            )[0]
            print_loc(global_print_loc)
            print_file_line(debug_print_file_line)

            if len(modelparams.shape) == 1:
                modelparams = modelparams[..., np.newaxis]
                
            times = expparams['t']

            if self.use_experimental_data == True:
                # sanity check that all times to be computed are available experimentally
                all_avail = np.all(
                    [
                    t in self.experimental_measurement_times
                    for t in times
                    ]
                )
                if all_avail == False:
                    print(
                        "[likelihood fnc]", 
                        "All times NOT available experimentally originally"
                    )

            if self.QLE is True:
                print_file_line(debug_print_file_line)
                try:
                    pr0 = Evo.get_pr0_array_qle(
                        t_list=times, 
                        modelparams=params,
                        oplist=operators, 
                        probe=probe, 
                        measurement_type=self.measurement_type,
                        growth_class = self.growth_class,  
                        use_experimental_data = self.use_experimental_data,
                        use_exp_custom=self.use_exp_custom,
                        exp_comparison_tol=self.exp_comparison_tol, 
                        enable_sparse = self.enable_sparse, 
                        log_file=self.log_file, 
                        log_identifier=self.log_identifier
                    )
                    print_file_line(debug_print_file_line)
                except:
                    log_print(
                        [
                            "[likelihood] failure to compute pr0",
                            "probe:", probe, 
                            "\n oplist:", operators
                        ]
                    )
                    print_file_line(debug_print_file_line)

            else: 
                # Built for IQLE but not in use/tested so unlikely to work. 
                pr0 = Evo.get_pr0_array_iqle(
                    t_list=times, 
                    modelparams=params,
                    oplist=operators, 
                    ham_minus=ham_minus, 
                    probe=probe,
                    use_exp_custom=self.use_exp_custom,
                    exp_comparison_tol=self.exp_comparison_tol, 
                    enable_sparse = self.enable_sparse, 
                    log_file=self.log_file, 
                    log_identifier=self.log_identifier
                )    

            if debug_log_print:
                log_print(
                    [
                    'Simulating experiment.',
                    'times:', times,
                    'len(outcomes):', len(outcomes),
                    '\nOutcomes:', outcomes, 
                    #'\n pr0:\n', pr0,
                    ],
                    self.log_file, 
                    self.log_identifier
                )

#        outcomes[[0]] = 1-outcomes[[0]]
        likelihood_array = (
            qi.FiniteOutcomeModel.pr0_to_likelihood_array(
                outcomes, pr0
            )
        )

        # if debug_log_print:
        #     log_print(
        #         [
        #         '\n likelihood values:\n:', likelihood_array
        #         ],
        #         self.log_file, 
        #         self.log_identifier
        #     )

        # if not times:
        #     times = [time]        
        # print("time:", times)            
        # print("pr0:", pr0)
        # print("likelihood array:", likelihood_array)
        return likelihood_array


