from __future__ import print_function  # so print doesn't show brackets

import numpy as np
import sys
import warnings
import copy

import scipy as sp
import qinfer as qi
import time

import qmla.shared_functionality.experimental_data_processing
import qmla.get_exploration_strategy
import qmla.memory_tests
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models
import qmla.logging 

global_print_loc = False
global debug_print
debug_print = False
global debug_mode
debug_mode = True
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
    This class, specified by an exploration strategy, defines how to 
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
    :param dict probes_system: set of probe states to be used during training
        for the system, indexed by (probe_id, num_qubits). 
    :param dict probes_simulator: set of probe states to be used during training
        for the simulator, indexed by (probe_id, num_qubits). Usually the same as 
        the system probes, but not always. 
    :param str exploration_rule: string corresponding to a unique exploration strategy,
        used to generate an explorationStrategy_ instance.
    :param dict experimental_measurements: fixed measurements of the target system, 
        indexed by time.
    :param list experimental_measurement_times: times indexed in experimental_measurements.
    :param str log_file: Path of log file.
    """

    ## INITIALIZER ##

    def __init__(
        self,
        model_name,
        model_constructor,
        true_model_constructor,
        num_probes,
        probes_system,
        probes_simulator,
        exploration_rules,
        experimental_measurements,
        experimental_measurement_times,
        log_file,
        qmla_id=-1, 
        evaluation_model=False,
        debug_mode=False,
        **kwargs
    ):

        # Essentials
        self.model_name = model_name
        self.model_constructor = model_constructor
        self.true_model_constructor = true_model_constructor
        self.true_hamiltonian = self.true_model_constructor.fixed_matrix

        # Instantiate QInfer Model class.
        super(QInferModelQMLA, self).__init__()

        # Infrastructure
        self.log_file = log_file
        self.qmla_id = qmla_id
        self.exploration_rules = exploration_rules
        self.probe_rotation_frequency = 10
        # TODO replace if want to use knowledge of initial signs:
        self.signs_of_inital_params = np.ones(self.n_modelparams) 

        # Exploration strategy
        try:
            self.exploration_class = qmla.get_exploration_strategy.get_exploration_class(
                exploration_rules=self.exploration_rules,
                log_file=self.log_file,
                qmla_id=self.qmla_id, 
            )
        except BaseException:
            self.log_print([
                "Could not instantiate exploration strategy {}. Terminating".format(
                    self.exploration_rules
                )
            ])
            raise

        # Required by QInfer: 
        self._min_freq = 0 # what does this do?
        self._solver = 'scipy'

        # How to use this model interface
        self.iqle_mode = self.exploration_class.iqle_mode 
        self.evaluation_model = evaluation_model
        
        self.log_print(["\nModel {} needs {} qubits. ".format(
            self.model_name,  self.model_constructor.num_qubits
        )])

        # TODO get experimental_measurements from exploration_class
        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_times = experimental_measurement_times

        # # Instantiate QInfer Model class.
        # super(QInferModelQMLA, self).__init__()

        try:
            self.probes_system = probes_system
            self.probes_simulator = probes_simulator
            self.probe_number = num_probes # TODO get from probe dict
        except:
            raise ValueError(
                "Probe dictionaries not passed to Qinfer model"
            )

        # Storage 
        self.store_likelihoods = {
            x : {} for x in ['system', 'simulator_median', 'simulator_mean']
        }
        self.likelihood_calls = {_ : 0 for _ in ['system', 'simulator']}
        self.summarise_likelihoods = {
            x : []
            for x in [
                'system', 
                'particles_median', 'particles_mean',
                'particles_std', 'particles_lower_quartile', 'particles_upper_quartile']
        }
        self.store_p0_diffs = []
        self.debug_mode = debug_mode
        self.timings = {
            'system': {}, 
            'simulator' : {}
        }
        for k in self.timings:
            self.timings[k] = {
                'expectation_values' : 0, 
                'get_pr0' : 0,
                'get_probe' : 0, 
                'construct_ham' : 0,
                'storing_output' : 0,
                'likelihood_array' : 0,
                'likelihood' : 0, 
            }
        self.calls_to_likelihood = 0 
        self.single_experiment_timings = {
            k : {} for k in ['system', 'simulator']
        }



    def log_print(
        self, 
        to_print_list, 
        log_identifier=None
    ):
        r"""Writng to unique QMLA instance log."""
        if log_identifier is None: 
            log_identifier = 'QInfer interface {}'.format(self.model_name)

        qmla.logging.print_to_log(
            to_print_list = to_print_list, 
            log_file = self.log_file, 
            log_identifier = log_identifier
        )

    def log_print_debug(
        self, 
        to_print_list
    ):
        r"""Log print if global debug_mode set to True."""

        if self.debug_mode:
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

        return self.model_constructor.num_terms

    @property
    def modelparam_names(self):
        r"""
        Returns the names of the various model parameters admitted by this
        model, formatted as LaTeX strings. (Inherited from Qinfer)
        """
        
        return self.model_constructor.terms_names


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

        # expparams are the {t, probe_id, w1, w2, ...} guessed parameters, i.e. each 
        # particle has a specific sampled value of the corresponding
        # parameter
        
        expnames = [
            ('t', 'float'),
            ('probe_id', 'int')
        ]
        
        for term in self.modelparam_names:
            expnames.append( (term, 'float') )

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
            * "simulate" an experiment (which can include outsourcing from here to perform a real experiment), 
            * update parameter distribution by comparing Np particles to the experimental result
        
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

        self.calls_to_likelihood+=1
        t_likelihood_start = time.time()
        super(QInferModelQMLA, self).likelihood(
            outcomes, modelparams, expparams
        )  # internal QInfer book-kepping

        # process expparams
        times = expparams['t'] # times to compute likelihood for. typicall only per experiment. 
        probe_id = expparams['probe_id'][0]
        expparams_sampled_particle = np.array(
            [expparams.item(0)[2:]]) # TODO THIS IS DANGEROUS - DONT DO IT OUTSIDE OF TESTS               
        self.log_print_debug([
            "expparams_sampled_particle:", expparams_sampled_particle
        ])
        self.ham_from_expparams = self.model_constructor.construct_matrix(
            expparams_sampled_particle
        )

        num_particles = modelparams.shape[0]
        num_parameters = modelparams.shape[1]

        # We assume that calls to likelihood are paired: 
        # one for system, one for simulator
        # therefore the same probe should be assumed for consecutive calls
        if num_particles == 1:
            # 1 particle indicates call to simulate_experiment 
            # => get system datum
            self.true_evolution = True
            timing_marker = "system"
        else:
            self.true_evolution = False
            timing_marker = "simulator"

        self.log_print_debug([
            "\n\nLikelihood fnc called. Probe counter={}. True system -> {}.".format(
                probe_id, self.true_evolution)
        ])

        # Get pr0, the probability of measuring the datum labelled '0'. 
        if self.true_evolution:
            t_init = time.time()
            probe = self.probes_system[
                probe_id,
                self.true_model_constructor.num_qubits,
            ]
            pr0 = self.get_system_pr0_array(
                times=times,
                probe=probe
            )
            self.timings[timing_marker]['get_pr0'] += time.time() - t_init
        else:
            t_init = time.time()
            probe = self.probes_simulator[
                probe_id,
                self.model_constructor.num_qubits
            ]

            pr0 = self.get_simulator_pr0_array(
                times=times,
                particles=modelparams,
                probe=probe,
            ) 
            self.timings[timing_marker]['get_pr0'] += time.time() - t_init
            
        # Convert pr0 probabilities to likelihoods for QInfer to use in updating distribution
        likelihood_array = (
            qi.FiniteOutcomeModel.pr0_to_likelihood_array(
                outcomes, pr0
            )
        )

        # Everything below here in this method is recording, no useful computation 
        # TODO probably most of it can be removed
        self.single_experiment_timings[timing_marker]['likelihood'] = time.time() - t_likelihood_start
        self.log_print_debug([
            '\ntrue_evo:', self.true_evolution,
            '\nevolution times:', times,
            '\nlen(outcomes):', len(outcomes),
            '\nprobe counter:', probe_id,
            '\nexp:', expparams,
            '\nOutcomes:', outcomes[:3],
            '\nparticles:', modelparams[:3],
            "\nPr0: ", pr0[:3], 
            "\nLikelihood: ", likelihood_array[0][:3],
            "\nexpparams_sampled_particle:", expparams_sampled_particle
        ])
        self.timings[timing_marker]['likelihood'] += time.time() - t_likelihood_start

        t_storage_start = time.time()
        if self.true_evolution: 
            self.log_print_debug(["Storing system likelihoods"])
            self.store_likelihoods['system'][self.likelihood_calls['system']] = pr0
            self.summarise_likelihoods['system'].append(np.median(pr0))
            self.likelihood_calls['system'] += 1 
        else:
            self.store_likelihoods['simulator_mean'][self.likelihood_calls['simulator']] = np.mean(pr0)
            self.store_likelihoods['simulator_median'][self.likelihood_calls['simulator']] = np.median(pr0)
            diff_p0 = np.abs( pr0 - self.store_likelihoods['system'][self.likelihood_calls['simulator']] )
            self.store_p0_diffs.append( [np.median(diff_p0), np.std(diff_p0)] )
            self.summarise_likelihoods['particles_mean'].append( np.median(pr0) )
            self.summarise_likelihoods['particles_median'].append( np.median(pr0) )
            self.summarise_likelihoods['particles_std'].append( np.std(pr0) )
            self.summarise_likelihoods['particles_lower_quartile'].append( np.percentile(pr0, 25) )
            self.summarise_likelihoods['particles_upper_quartile'].append( np.percentile(pr0, 75) )
            self.likelihood_calls['simulator'] += 1 
        self.single_experiment_timings[timing_marker]['storage'] = time.time() - t_storage_start
        self.log_print_debug([
            "Setting single_experiment_timings for {}[{}] -> {}".format(
                timing_marker, 'storage', time.time() - t_storage_start
            )
        ])

        self.log_print_debug(["Stored likelihoods"])

        if self.evaluation_model:
            self.log_print_debug([
                "\nSystem evolution {}. t={} Likelihood={}".format(
                self.true_evolution, times[0], likelihood_array[:3]
            )])
        
        return likelihood_array

    def get_system_pr0_array(
        self, 
        times,
        probe
    ):
        r"""
        Compute pr0 array for the system. 
        # TODO compute e^(-iH) once for true Hamiltonian and use that rather than computing every step. 

        For user specific data, or method to compute system data, replace this function 
            in exploration_strategy.qinfer_model_subroutine. 

        :param list times: times to compute pr0 for; usually single element.
        
        :returns np.ndarray pr0: probabilities of measuring specified outcome on system
        """
        from rq import timeouts

        hamiltonian = self.true_model_constructor.fixed_matrix
        t_init = time.time()

        if self.iqle_mode: 
            # TODO use different fnc for IQLE
            hamiltonian -= self.ham_from_expparams

        if np.any(np.isnan(hamiltonian)):
            self.log_print([
                "NaN detected in Hamiltonian. Ham from expparams:", 
                self.ham_from_expparams
            ])

        # compute likelihoods
        probabilities = []
        for t in times:

            t_init = time.time()
            prob_meas_input_state = self.exploration_class.get_expectation_value(
                ham=hamiltonian,
                t=t,
                state=probe,
                log_file=self.log_file,
                log_identifier='get pr0 call exp val'
            )
            self.timings['system']['expectation_values'] += time.time() - t_init
            probabilities.append(prob_meas_input_state)

        pr0 = np.array([probabilities])
        return pr0

    def get_simulator_pr0_array(
        self, 
        particles, 
        times,
        probe
    ):
        r"""
        Compute pr0 array for the simulator. 

        For user specific data, or method to compute simulator data, 
            replace this function 
            in exploration_strategy.qinfer_model_subroutine. 
        Here we pass the candidate model's operators and particles
            to default_pr0_from_modelparams_times_.

        :param list times: times to compute pr0 for; usually single element.
        :param np.ndarry particles: list of particles (parameter-lists), used to construct
            Hamiltonians. 
        
        :returns np.ndarray pr0: probabilities of measuring specified outcome
        """
       
        num_particles = len(particles)
        pr0 = np.empty([num_particles, len(times)])

        for particle_idx in range(num_particles):  
            # loop over particles
            if self.evaluation_model:
                hamiltonian = self.model_constructor.fixed_matrix
            else:
                hamiltonian = self.model_constructor.construct_matrix(
                    particles[particle_idx]
                )
            self.log_print_debug([
                "Hamiltonian from model constructor:", 
                hamiltonian
            ])
            time_idx = -1
            for t in times:
                time_idx += 1 # TODO cleaner way of indexing pr0 array
                prob_meas_input_state = self.exploration_class.get_expectation_value(
                    ham=hamiltonian,
                    t=t,
                    state=probe,
                    log_file=self.log_file,
                    log_identifier='get pr0 call exp val'
                )
                pr0[particle_idx][time_idx] = prob_meas_input_state

        return pr0


class QInferNVCentreExperiment(QInferModelQMLA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_system_pr0_array(
        self, 
        times,
        particles, 
        **kwargs
    ):
        self.log_print_debug(["Getting pr0 from experimental dataset."])
        # time = expparams['t']
        if len(times) > 1:
            self.log_print("Multiple times given to experimental true evolution:", times)
            sys.exit()

        time = times[0]
        
        try:
            # If time already exists in experimental data
            experimental_expec_value = self.experimental_measurements[time]
        except BaseException:
            # map to nearest experimental time
            try:
                experimental_expec_value = qmla.shared_functionality.experimental_data_processing.nearest_experimental_expect_val_available(
                    times=self.experimental_measurement_times,
                    experimental_data=self.experimental_measurements,
                    t=time
                )
            except:
                self.log_print_debug([
                    "Failed to get experimental data point"
                ])
                raise
            self.log_print_debug([
                "experimental value for t={}: {}".format(
                    time, 
                    experimental_expec_value
                )
            ])
        self.log_print_debug([
            "Using experimental time", time,
            "\texp val:", experimental_expec_value
        ])
        pr0 = np.array([[experimental_expec_value]])
        self.log_print_debug([
            "pr0 for system:", pr0
        ])
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


class QInferInterfaceJordanWigner(QInferModelQMLA):
    r"""
    For use when models are implemented via Jordan Wigner transformation, 
    since this invokes 2 qubits per site in the system. 
    Therefore, everything remains as in other models, 
    apart from probe selection should use the appropriate probe id, 
    but twice the number of qubits specified by the model. 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_probe(
        self, 
        probe_id, 
        probe_set
    ):
        self.log_print([
            "Using JW get_probe"
        ])
        if probe_set == 'simulator':
            probe = self.probes_simulator[
                probe_id,
                2*self.model_constructor.num_qubits
            ]
            return probe

        elif probe_set == 'system': 
            # get dimension directly from true model since this can be generated by another ES 
            # and therefore note require the 2-qubit-per-site overhead of Jordan Wigner.
            dimension = np.log2(np.shape(self.true_hamiltonian)[0])
            probe = self.probes_system[
                probe_id,
                self.true_model_constructor.num_qubits
            ]
            return probe
        else:
            self.log_print([
                "get_probe must either act on simulator or system, received {}".format(probe_set)
            ])
            raise ValueError(
                "get_probe must either act on simulator or system, received {}".format(probe_set)
            )


class QInferInterfaceAnalytical(QInferModelQMLA):
    r"""
    Analytically computes the likleihood for an exemplary case. 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_system_pr0_array(
        self, 
        times,
        particles, 
    ):

        pr0 = np.empty([len(particles), len(times)])
        t = times[0]
        self.log_print_debug([
            "(sys) particles:", particles,
            "time: ", t,
            "\n shapes: prt={} \t times={}".format(np.shape(particles), np.shape(times))
        ])

        for evoId in range(len(particles)):
            particle = particles[evoId][0]
            for t_id in range(len(times)):
                pr0[evoId][t_id] = (np.cos(particle * t / 2))**2

        return pr0

    def get_simulator_pr0_array(
        self, 
        particles, 
        times,
        # **kwargs
    ):
        pr0 = np.empty([len(particles), len(times)])
        t = times[0]
        self.log_print_debug([
            "(sim) particles:", particles,
            "time: ", t,
            "\n shapes: prt={} \t times={}".format(np.shape(particles), np.shape(times))
        ])

        for evoId in range(len(particles)):
            particle = particles[evoId]  
            for t_id in range(len(times)):
                pr0[evoId][t_id] = (np.cos(particle * t / 2))**2

        return pr0

