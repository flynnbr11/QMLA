from __future__ import print_function  # so print doesn't show brackets

import numpy as np
import sys
import warnings
import copy

import scipy as sp
import qinfer as qi

import qmla.shared_functionality.experimental_data_processing
import qmla.get_growth_rule
import qmla.memory_tests
import qmla.shared_functionality.probe_set_generation
import qmla.database_framework
import qmla.logging 

global_print_loc = False
global debug_print
debug_print = False
global debug_log_print
debug_log_print = False
global debug_print_file_line
debug_print_file_line = False


class QInferModelQMLA(qi.FiniteOutcomeModel):
    r"""
    Interface between QMLA and QInfer.

    QInfer is a library for performing Bayesian inference
        on quantum data for parameter estimation.
        It underlies the Quantum Hamiltonian Learning subroutine
        employed within QMLA.
        Bayesian inference relies on comparisons likelihoods
        of the target and candidate system. 
    This class, specified by a growth rule, defines how to 
        compute the likelihood for the user's system. 
        Most functionality is inherited from QInfer, but methods listed 
        here are edited for QMLA's needs. 
    The likelihood function given here should suffice for most QMLA 
        implementations, though users may want to overwrite 
        get_system_pr0_array and get_simulator_pr0_array, 
        for instance to specify which experimental data points to use. 
    
    :param str model_name: Unique string representing a model.
    :param np.ndarray modelparams: list of parameters to multiply by operators, 
        unused for QMLA reasons but required by QInfer. 
    :param np.ndarray oplist: Set of operators whose sum
        defines the evolution Hamiltonian 
        (where each operator is associated with a distinct parameter).
    :param np.ndarray true_oplist: list of operators of the target system,
        used to construct true hamiltonian.
    :param np.ndarray trueparams: list of parameters of the target system,
        used to construct true hamiltonian.
    :param int num_probes: number of probes available in the probe sets, 
        used to loop through probe set
    :param dict probe_dict: set of probe states to be used during training
        for the system, indexed by (probe_id, num_qubits). 
    :param dict sim_probe_dict: set of probe states to be used during training
        for the simulator, indexed by (probe_id, num_qubits). Usually the same as 
        the system probes, but not always. 
    :param str growth_generator: string corresponding to a unique growth rule,
        used to generate a GrowthRule_ instance.
    :param dict experimental_measurements: fixed measurements of the target system, 
        indexed by time.
    :param list experimental_measurement_times: times indexed in experimental_measurements.
    :param str log_file: Path of log file.
    """

    ## INITIALIZER ##

    def __init__(
        self,
        model_name,
        modelparams,
        oplist,
        true_oplist,
        truename,
        trueparams,
        num_probes,
        probe_dict,
        sim_probe_dict,
        growth_generation_rule,
        experimental_measurements,
        experimental_measurement_times,
        log_file,
        **kwargs
    ):
        self._oplist = oplist
        self._a = 0
        self._b = 0
        self._modelparams = modelparams
        self.signs_of_inital_params = np.sign(modelparams)
        self._true_oplist = true_oplist
        self._trueparams = trueparams
        self._truename = truename
        self._true_dim = qmla.database_framework.get_num_qubits(self._truename)
        # self.use_experimental_data = use_experimental_data
        self.log_file = log_file
        self.growth_generation_rule = growth_generation_rule
        try:
            self.growth_class = qmla.get_growth_rule.get_growth_generator_class(
                growth_generation_rule=self.growth_generation_rule,
                # use_experimental_data=self.use_experimental_data,
                log_file=self.log_file
            )
        except BaseException:
            self.log_print(
                [
                    "Could not instantiate growth rule {}. Terminating".foramt(
                        self.growth_generation_rule
                    )
                ]
            )
        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_times = experimental_measurement_times
        # Required by QInfer: 
        self._min_freq = 0 # what does this do?
        self._solver = 'scipy'
        # This is the solver used for time evolution scipy is faster
        # QuTip can handle implicit time dependent likelihoods

        self.model_name = model_name
        self.model_dimension = qmla.database_framework.get_num_qubits(self.model_name)
        self.inBayesUpdates = False
        if true_oplist is not None and trueparams is None:
            raise(
                ValueError(
                    '\nA system Hamiltonian with unknown \
                    parameters was requested'
                )
            )
        super(QInferModelQMLA, self).__init__(self._oplist)

        try:
            self.probe_dict = probe_dict
            self.sim_probe_dict = sim_probe_dict
            self.probe_number = num_probes
        except:
            raise ValueError(
                "Probe dictionaries not passed to Qinfer model"
            )

    def log_print(
        self, 
        to_print_list, 
        log_identifier=None
    ):
        r"""Writng to unique QMLA instance log."""
        if log_identifier is None: 
            log_identifier = 'QInfer interface'

        qmla.logging.print_to_log(
            to_print_list = to_print_list, 
            log_file = self.log_file, 
            log_identifier = log_identifier
        )

    def log_print_debug(
        self, 
        to_print_list
    ):
        r"""Log print if global debug_log_print set to True."""

        if debug_log_print:
            self.log_print(
                to_print_list = to_print_list,
                log_identifier = 'QInfer interface debug'
            )

    ## PROPERTIES ##
    @property
    def n_modelparams(self):
        r"""
        Number of parameters in the specific model 
        typically, in QMLA, we have one parameter per model.
        """

        return len(self._oplist)

    @property
    def modelparam_names(self):
        r"""
        Returns the names of the various model parameters admitted by this
        model, formatted as LaTeX strings. (Inherited from Qinfer)
        """

        modnames = ['w0']
        for modpar in range(self.n_modelparams - 1):
            modnames.append('w' + str(modpar + 1))
        return modnames


    @property
    def expparams_dtype(self):
        r"""
        Returns the dtype of an experiment parameter array. 
        
        For a model with single-parameter control, this will likely be a scalar dtype,
        such as ``"float64"``. More generally, this can be an example of a
        record type, such as ``[('time', py.'float64'), ('axis', 'uint8')]``.
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        In the context of QMLA the expparams_dtype are assumed to be a list of tuple where
        the first element of the tuple identifies the parameters (including type) while the second element is
        the actual type of of the parameter, typicaly a float.
        (Modified from Qinfer).
        """

        # expparams are the {t, w1, w2, ...} guessed parameters, i.e. each 
        # particle has a specific sampled value of the corresponding
        # parameter
        # 
        expnames = [('t', 'float')]
        for exppar in range(self.n_modelparams):
            expnames.append(('w_' + str(exppar + 1), 'float'))
        return expnames

    ################################################################################
    # Methods
    ################################################################################

    def are_models_valid(self, modelparams):
        r"""
        Checks that the proposed models are valid.

        Before setting new distribution after resampling,
        checks that all parameters have same sign as the
        initial given parameter for that term.
        Otherwise, redraws the distribution.
        Modified from qinfer.
        """

        same_sign_as_initial = False
        if same_sign_as_initial == True:
            new_signs = np.sign(modelparams)
            validity_by_signs = np.all(
                np.sign(modelparams) == self.signs_of_inital_params,
                axis=1
            )
            return validity_by_signs
        else:
            validity = np.all(np.abs(modelparams) > self._min_freq, axis=1)
            return validity

    def n_outcomes(self, expparams):
        r"""
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
        r"""
        Function to calculate likelihoods for all the particles
        
        Inherited from Qinfer:
            Calculates the probability of each given outcome, conditioned on each
            given model parameter vector and each given experimental control setting.

        QMLA modifications: 
            Given a list of experiments to perform, expparams, 
            extract the time list. Typically we use a single experiment
            (therefore single time) per update.
            QInfer passes particles as modelparams.
            QMLA updates its knowledge in two steps:
                * "simulate" an experiment 
                    (which can include outsourcing from here to perform a real experiment), 
                * update parameter distribution 
                    by comparing Np particles to the experimental result
            It is important that the comparison is fair, meaning:
                * The evolution time must be the same
                * The probe state to evolve must be the same.

            To simulate the experiment, we call QInfer's simulate_experiment,
                which calls likelihood(), passing a single particle. 
            The update function calls simulate_experiment with Np particles. 
            Therefore we know, when a single particle is passed to likelihood, 
                that we want to call the true system (we know the true parameters 
                and operators by the constructor of this class). 
            So, when a single particle is detected, we circumvent QInfer by triggering
                get_system_pr0_array. Users can overwrite this function as desired; 
                by default it computes true_hamiltonian, 
                and computes the likelhood for the given time. 
            When >1 particles are detected, pr0 is computed by constructing Np 
                candidate Hamiltonians, each corresponding to a single particle, 
                where particles are chosen by Qinfer and given as modelparams.
                This is done through get_simulator_pr0_array.
            We know calls to likelihood are coupled: 
                one call for the system, and one for the update, 
                which must use the same probes. Therefore probes are indexed
                by a probe_id as well as their dimension. 
                We track calls to likelihood() in _a and increment the probe_id
                to pull every second call, to ensure the same probe_id is used for 
                system and simulator.

        :param np.ndarray outcomes: outcomes of the experiments
        :param np.ndarray modelparams: 
            values of the model parameters particles 
            A shape ``(n_particles, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        
        :param np.ndarray expparams: 
            experimental parameters, 
            A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Simulatable.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
            
        :rtype: np.ndarray
        :return: A three-index tensor ``L[i, j, k]``, where ``i`` is the outcome
            being considered, ``j`` indexes which vector of model parameters was used,
            and where ``k`` indexes which experimental parameters where used.
            Each element ``L[i, j, k]`` then corresponds to the likelihood
            :math:`\Pr(d_i | \vec{x}_j; e_k)`.
        """

        
        super(QInferModelQMLA, self).likelihood(
            outcomes, modelparams, expparams
        )  # just adds to self._call_count (Qinfer abstact model class)
        times = expparams['t'] # times to compute likelihood for. typicall only per experiment. 
        num_particles = modelparams.shape[0]
        num_parameters = modelparams.shape[1]
        # assumption is that calls to likelihood are paired: 
        # one for system, one for simulator
        # therefore the same probe should be assumed for consecutive calls
        # probe id is tracked with _a and _b.
        # i.e. increments each 2nd call, loops back when probe dict exhausted
        self._a += 1
        if self._a % 2 == 1:
            self._b += 1
        self.probe_counter = (self._b % int(self.probe_number)) 


        if num_particles == 1:
            # TODO better mechanism to determine if true_evo, 
            # rather than assuming 1 particle => system
            # call the system, use the true paramaters as a single particle, 
            # to get the true evolution
            true_evo = True
            params = [copy.deepcopy(self._trueparams)]
        else:
            true_evo = False
            params = modelparams

        try:
            if true_evo:
                pr0 = self.get_system_pr0_array(
                    times=times,
                    particles=params,
                )
            else:
                pr0 = self.get_simulator_pr0_array(
                    times=times,
                    particles=params,
                ) 
        except:
            self.log_print(
                [
                    "Failed to compute pr0.",
                ]
            )
            sys.exit()

        likelihood_array = (
            qi.FiniteOutcomeModel.pr0_to_likelihood_array(
                outcomes, pr0
            )
        )
        self.log_print_debug(
            [
                'Simulating experiment.',
                'times:', times,
                'true_evo:', true_evo,
                'len(outcomes):', len(outcomes),
                '_a = {}, _b={}'.format(self._a, self._b),
                'probe counter:', self.probe_counter,
                '\nexp:', expparams,
                '\nOutcomes:', outcomes,
                '\nmodelparams:', params,
            ]
        )
        self.log_print_debug(
            [
                "Outcomes: ", outcomes, 
                "\nPr0: ", pr0, 
                "\nLikelihood: ", likelihood_array
            ]
        )

        return likelihood_array

    def get_system_pr0_array(
        self, 
        times,
        particles, 
        # **kwargs
    ):
        r"""
        Compute pr0 array for the system. 

        For user specific data, or method to compute system data, replace this function 
            in growth_rule.qinfer_model_class. 
        Here we pass the true operator list and true parameters to 
            default_pr0_from_modelparams_times_.

        :param list times: times to compute pr0 for; usually single element.
        :param np.ndarry particles: list of parameter-lists, used to construct
            Hamiltonians. In this case, there should be a single particle
            corresponding to the true parameters. 
        
        :returns np.ndarray pr0: probabilities of measuring specified outcome
        """

        operator_list = self._true_oplist
        ham_num_qubits = self._true_dim
        # format of probe dict keys: (probe_id, qubit_number)
        # probe_counter controlled in likelihood method
        probe = self.probe_dict[
            self.probe_counter,
            ham_num_qubits
        ]
        # TODO: could just work with true_hamiltonian, worked out on __init__
        return self.default_pr0_from_modelparams_times(
            t_list = times,
            particles = particles, 
            oplist = operator_list, 
            probe = probe, 
            # **kwargs
        )

    def get_simulator_pr0_array(
        self, 
        particles, 
        times,
        # **kwargs
    ):
        r"""
        Compute pr0 array for the simulator. 

        For user specific data, or method to compute simulator data, replace this function 
            in growth_rule.qinfer_model_class. 
        Here we pass the candidate model's operators and particles
            to default_pr0_from_modelparams_times_.

        :param list times: times to compute pr0 for; usually single element.
        :param np.ndarry particles: list of particles (parameter-lists), used to construct
            Hamiltonians. 
        
        :returns np.ndarray pr0: probabilities of measuring specified outcome
        """
        ham_num_qubits = self.model_dimension
        # format of probe dict keys: (probe_id, qubit_number)
        # probe_counter controlled in likelihood method
        probe = self.sim_probe_dict[
            self.probe_counter,
            ham_num_qubits 
        ]
        operator_list = self._oplist
        return self.default_pr0_from_modelparams_times(
            t_list = times, 
            particles = particles, 
            oplist = operator_list, 
            probe = probe, 
            # **kwargs
        )

    def default_pr0_from_modelparams_times(
        self,
        t_list,
        particles,
        oplist,
        probe,
        **kwargs
    ):
        r"""
        Compute probabilities of available outputs as an array.

        :param np.ndarray t_list: 
            List of times on which to perform experiments
        :param np.ndarray particles: 
            values of the model parameters particles 
            A shape ``(n_particles, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param list oplist:
            list of the operators defining the model
        :param np.ndarray probe: quantum state to evolve

        :returns np.ndarray pr0: list of probabilities (one for each particle).
            The calculation, meaning and interpretation of these probabilities 
            depends on the user defined GrowthRule.expectation_value function. 
            By default, it is the expecation value:
                | < probe.transpose | e^{-iHt} | probe > |**2,
                but can be replaced in the GrowthRule_.  
        """

        from rq import timeouts
        self.log_print_debug(
            [
                "Probe[0] (dimension {}): \n {}".format(
                    np.shape(probe),
                    probe[0],
                ),
                "Times: ", t_list
            ]
        )

        num_particles = len(particles)
        num_times = len(t_list)
        output = np.empty([num_particles, num_times])

        for evoId in range(num_particles):  
            try:
                ham = np.tensordot(
                    particles[evoId], oplist, axes=1
                )
            except BaseException:
                self.log_print(
                    [
                        "Failed to build Hamiltonian.",
                        "\nparticles:", particles[evoId],
                        "\noplist:", oplist
                    ],
                )
                raise
            for tId in range(len(t_list)):
                t = t_list[tId]
                if t > 1e6:  # Try limiting times to use to 1 million
                    import random
                    # random large number but still computable without error
                    t = random.randint(1e6, 3e6)
                try:
                    likel = self.growth_class.expectation_value(
                        ham=ham,
                        t=t,
                        state=probe,
                        log_file=self.log_file,
                        log_identifier='get pr0 call exp val'
                    )
                    output[evoId][tId] = likel

                except NameError:
                    self.log_print(
                        [
                            "Error raised; unphysical expecation value.",
                            "\nHam:\n", ham,
                            "\nt=", t,
                            "\nState=", probe,
                        ],
                    )
                    sys.exit()
                except timeouts.JobTimeoutException:
                    self.log_print(
                        [
                            "RQ Time exception. \nprobe=",
                            probe,
                            "\nt=", t, "\nHam=",
                            ham
                        ],
                    )
                    sys.exit()

                if output[evoId][tId] < 0:
                    print("NEGATIVE PROB")
                    self.log_print(
                        [
                            "[QLE] Negative probability : \
                            \t \t probability = ",
                            output[evoId][tId]
                        ],
                    )
                elif output[evoId][tId] > 1.001:
                    self.log_print(
                        [
                            "[QLE] Probability > 1: \
                            \t \t probability = ",
                            output[evoId][tId]
                        ]
                    )
        return output



class QInferNVCentreExperiment(QInferModelQMLA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_system_pr0_array(
        self, 
        times,
        particles, 
        **kwargs
    ):
        # time = expparams['t']
        if len(times) > 1:
            self.log_print("Multiple times given to experimental true evoluation:", times)
            sys.exit()

        time = times[0]
        # self.log_print(
        #     [
        #         'Getting system outcome',
        #         'time:\n', time
        #     ]
        # )
        
        try:
            # If time already exists in experimental data
            experimental_expec_value = self.experimental_measurements[time]
        except BaseException:
            # map to nearest experimental time
            self.log_print_debug(
                [
                    "In except.",
                    # "exp times: \n{} \n exp meas: {}".format(
                    #     self.experimental_measurement_times, 
                    #     self.experimental_measurements
                    # )
                ]
            )
            try:
                experimental_expec_value = qmla.shared_functionality.experimental_data_processing.nearest_experimental_expect_val_available(
                    times=self.experimental_measurement_times,
                    experimental_data=self.experimental_measurements,
                    t=time
                )
            except:
                self.log_print_debug(
                    [
                        "Failed to get experimental data point"
                    ]
                )
                raise
            self.log_print_debug(
                [
                    "experimental value for t={}: {}".format(
                        time, 
                        experimental_expec_value
                    )
                ]
            )
        self.log_print_debug(
            [
                "Using experimental time", time,
                "\texp val:", experimental_expec_value
            ],
        )
        pr0 = np.array([[experimental_expec_value]])
        self.log_print_debug(
            [
                "pr0 for system:", pr0
            ]
        )
        return pr0

    def get_simulator_pr0_array(
        self, 
        particles, 
        times,
        # **kwargs
    ):
        # map times to experimentally available times
        mapped_times = [
            qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(
                times = self.experimental_measurement_times,
                t = t
            )
            for t in times
        ]
        return super().get_simulator_pr0_array(
            particles, 
            mapped_times
        )


