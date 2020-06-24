from __future__ import absolute_import
import sys
import os
import pickle
import numpy as np
import itertools

import qmla.shared_functionality.prior_distributions
import qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.probe_set_generation as probe_set_generation
import qmla.shared_functionality.expectation_values
import qmla.utilities
import qmla.construct_models as construct_models
import qmla.growth_rules.rating_system
import qmla.shared_functionality.qinfer_model_interface
from qmla.growth_rules.growth_rule_decorator import GrowthRuleDecorator

__all__ = [
    'GrowthRule'
]

class GrowthRule():
    r"""
    User defined mechanism to control which models are considered by QMLA. 

    By changing the attributes, various aspects of QMLA are altered. 
    A number of growth rule attributes point to standalone methods available within QMLA, 
    e.g. to generate probes according to a desired mechanism. 
    This allows the user to easily  change functionality in a modular fashion.
    To develop a new growth rule, users should read the definitions of all 
    growth rule attributes listed in the various ``setup`` methods, and ensure
    that the default are suitable for their system, or that they have replaced them
    in their custom growth rule. 
    The ``setup`` methods are:
    
    * :meth:`~qmla.growth_rules.GrowthRule._setup_modular_functions`
    * :meth:`~qmla.growth_rules.GrowthRule._setup_true_model`
    * :meth:`~qmla.growth_rules.GrowthRule._setup_model_learning`    
    * :meth:`~qmla.growth_rules.GrowthRule._setup_tree_infrastructure`
    * :meth:`~qmla.growth_rules.GrowthRule._setup_logistics`

    """

    # superclass for growth generation rules
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):

        self.growth_generation_rule = growth_generation_rule

        if 'log_file' in kwargs:
            self.log_file = kwargs['log_file']
        else:
            self.log_file = '.default_qmla_log.log'
        
        if 'true_params_path' in kwargs: 
            self.true_params_path = kwargs['true_params_path']
        else: 
            self.true_params_path = None
        
        if 'plot_probes_path' in kwargs: 
            self.plot_probes_path = kwargs['plot_probes_path']
        else: 
            self.plot_probes_path = None

        # self.assign_parameters()
        self._setup_modular_functions()
        self._setup_true_model()
        self._setup_model_learning()
        self._setup_tree_infrastructure()
        self._setup_logistics()

    ##########
    # Section: Set up, assign parameters etc
    ##########

    def _setup_modular_functions(self):
        r"""
        Assign modular functions for the realisation of this growth rule.

        These are called by wrappers in the parent :class:`~qmla.GrowthRule` class; 
        the wrapper methods are called throughout a QMLA instance, e.g. to generate a set of probes
        according to the requirements of the user's growth rule. 
        Note also that these wrappers can be directly over written in a user growth rule, or more simply 
        the functionality can be replaced by adding a standalone function to ``qmla.shared_functionality``, 
        and replacing the methods in the user growth rule's ``__init__`` method.
        The wrappers and corresponding class attributes are:

        * :meth:`~qmla.growth_rules.GrowthRule.expectation_value` : ``expectation_value_function``
        * :meth:`~qmla.growth_rules.GrowthRule.generate_probes` : ``probe_generation_function``
        * :meth:`~qmla.growth_rules.GrowthRule.plot_probe_generator` : ``plot_probe_generation_function``
        * :meth:`~qmla.growth_rules.GrowthRule.heuristic` : ``model_heuristic_function``
        * :meth:`~qmla.growth_rules.GrowthRule.qinfer_model` : ``qinfer_model_class``
        * :meth:`~qmla.growth_rules.GrowthRule.get_prior` : ``prior_distribution_generator``
        * :meth:`~qmla.growth_rules.GrowthRule.latex_name` : ``latex_model_naming_function``

        """

        # Measurement
        self.expectation_value_function = qmla.shared_functionality.expectation_values.probability_from_default_expectation_value

        # Probes
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = True  # i.e. system and simulator get same probes for learning
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.probe_noise_level = 0 # 1e-5

        # Experiment design
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
                
        # QInfer interface
        self.qinfer_model_class = qmla.shared_functionality.qinfer_model_interface.QInferModelQMLA

        # Prior distribution
        self.prior_distribution_generator = qmla.shared_functionality.prior_distributions.gaussian_prior

        # Map model name strings to latex representation
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.pauli_set_latex_name


    def _setup_true_model(self):
        r"""
        Target system data, such as  model and parameters.

        * Set up * 
        
        true_model 
            target model for QMLA; presumed model for QHL
        qhl_models
            for multi-model QHL mode, which models to learn parameters of
        true_model_terms_params
            target parameters for terms in true model 
            (assigned randomly if not explicitly set here)
            e.g. ``{ 
            'pauliSet_1_x_d1' : 0.75,
            'pauliSet_1_x_d1' : 0.25
            }``

        """
        self.true_model = 'pauliSet_1_x_d1'
        self.qhl_models = ['pauliSet_1_x_d1', 'pauliSet_1_y_d1', 'pauliSet_1_z_d1']
        self.true_model_terms_params = {}

        # Set or retrieve system data shared over instances
        self.get_true_parameters()

    def _setup_model_learning(self):
        r"""
        Parameters used on the model learning level.

        *Setting prior distribution*

        gaussian_prior_means_and_widths: 
            Starting mean and width
            for normal prior distribution used for each parameter, 
            e.g. ``{ pauliSet_1_x_d1 : (0.5, 0.2), 'pauliSet_1_y_d1' : (0.5, 0.2)}``
        
        *True parameter selection*

        min_param, max_param
            used to generate  a 
            normal distribution  :math:`N(\mu, \sigma)` 
            where :math:`\mu=` ``mean(min_param, max_param)``, 
            :math:`\sigma=` 
            ``(max_param - min_param) / 4 ``, 
            which is then used to generate true parameters 
            for each parameter which is not explicitly set in 
            ``true_model_terms_params``, 
            and also as the span of the prior distribution 
            for each parameter not  explicitly set in 
            ``gaussian_prior_means_and_widths``. 
            See :meth:`~qmla.growth_rules.GrowthRule.get_prior` for details.
        true_param_cov_mtx_widen_factor
            when selecting true parameters, 
            they are chosen from :math:`N` defined for ``min_param``, ``max_param``; 
            this factor :math:`k` changes the distribution to :math:`N(\mu, k \sigma)` for the selection
            of the true parameters. e.g. so that true parameters are further than :math:`1 \sigma`
            from the starting prior, to ensure the algorithm is robust to such cases. 
        prior_random_mean
            if True, overwrites any true parameter set in ``true_model_terms_params``
            with a randomly drawn parameter as outlined above.

        *Learning*

        num_probes
            number of probes generated for the probe dictionary, 
            which are cycled over during parameter learning and model comparison.
        max_time_to_consider
            Upper limit on time. 
                1. used for all plots of dynamics
                2. given to the experiment design heuristic which may use it upper-bound
                experimental times chosen. 
        terminate_learning_at_volume_convergence
            Whether to stop learning when a model reaches a given threshold in volume
        volume_convergence_threshold
            The volume at which to terminate learning if ``terminate_learning_at_volume_convergence==True`` 
        iqle_mode
            True for interactive quantum likelihood estimation; 
            False for quantum likelihood estimation. 
            i.e. the method of parmeater learning.
            Note IQLE is far stronger for learning, but is not available for physical systems in general, 
            since it assumes access to a coherent quantum channel which maps the target system 
            to a simulator. 
        qinfer_resampler_threshold
            :math:`k_r`, fraction of particles below which to trigger a resampling event. 
            i.e. when the effective sample size is less than this fraction of the initial number of particles, 
            :math:`N_{ESS} < t=k_r N_p`, the QInfer updater resamples the distribution. 
        qinfer_resampler_a
            `a` parameter used by Liu-West resampler within 
            `QInfer SMCUpdater <http://docs.qinfer.org/en/latest/guide/smc.html?highlight=smcupdater>`_.
        hard_fix_resample_effective_sample_size
            absolute number of effective particles, below which
            to trigger resample. 
        reallocate_resources`` : 
            whether to decrease/increase the number of experiments/particles
            used to train models based on their relative complexity, compared with an assumed maximally 
            complicated model given by ``max_num_parameter_estimate``. 
            e.g. you make 10 000 particles available, but only want to invoke such a high cost for 
            models with 6 parameters, and simpler models could learn with proportionally less particles. 
        fraction_particles_for_bf 
            fraction of particles to use during pairwise comparison between models. 
            e.g. if 10 000 particles are used for parameter learning for each model, 
            the Bayes factor from 1 000 particles is expected to be in favour of the same 
            model as the Bayes factor using 10 000 particles, using far less time, but with weaker evidence. 
        fraction_experiments_for_bf
            fraction of experiments to use during pairwise comparison between models. 
            In particular, the latter portion of experiments are used. 
            This is equivalent to allowing a number of `burn-in` experiments, which are not counted 
            towards the Bayes factor. 
            # TODO BF(A,B) uses full renormalisation_record of A, and only updates the portion of B's expeiriments; 
            it should base comparison on latter fraction only. 

        *Plotting*

        plot_time_increment
             :math:`\Delta t` between each point plotted in dynamics plots. 


        """

        # Setting prior distribution
        self.gaussian_prior_means_and_widths = {}

        # True parameter selection
        self.min_param = 0
        self.max_param = 1
        self.true_param_cov_mtx_widen_factor = 1
        self.prior_random_mean = False
        self.fixed_true_terms = False        

        # Learning
        self.num_probes = 40
        self.max_time_to_consider = 15  # arbitrary time units
        self.terminate_learning_at_volume_convergence = False
        self.volume_convergence_threshold = 1e-8
        self.iqle_mode = False
        self.reallocate_resources = False
        self.max_num_parameter_estimate = 2
        self.qinfer_resampler_a = 0.98
        self.qinfer_resampler_threshold = 0.5
        self.hard_fix_resample_effective_sample_size = None
        self.fraction_experiments_for_bf = 1.0
        self.fraction_particles_for_bf = 1.0 # testing whether reduced num particles for BF can work 

        # Plotting
        self.plot_time_increment = None


    def _setup_tree_infrastructure(self):
        r"""
        Determining how the GR tree grows, when it should stop etc. 

        *Tree development*

        initial_models
            models to place on the first branch of the 
            :class:`~qmla.GrowthRuleTree` corresponding to this growth rule. 
        tree_completed_initially
            if True, no spawning stage is performed
        prune_completed_initially
            if True, no pruning stage is performed
            # TODO review how pruning attributes are called/checked
            # TODO improve docs about pruning
        max_spawn_depth
            Number of branches to spawn for this GR
        max_num_qubits
            Maximum number of qubits expected in any model entertained by 
            this GR. Used to generate probes up to this dimension. 
        max_num_probe_qubits
            TODO remove: serves same purpose as max_num_qubtis
        num_top_models_to_build_on
            number of models used during construction of next set of models
            by :meth:`~qmla.growth_rules.GrowthRule.generate_models`. 
            Mostly unused by more-recent GRs. 


        *Ratings class*
        
        ratings_class
            scheme for rating models based on pairwise comparisons.
            Not used by default; used in genetic algorithm to determine top models. 


        Comparisons within branches:

        branch_champion_selection_strategy
            (string) mechanism by which to decide 
            within a layer/branch, which model is favoured. 
                - ``number_comparison_wins`` (default) : number of pairwise wins of each model
                - ``ratings`` - models ranked by their rating as determined by the ``ratings_class``.
        branch_comparison_strategy 
            mechanism by which to perform pairwise model comparisons.
                - ``all`` (default): completely connected graph
                - ``optimal_graph``: generate a partially connected graph
        

        *Champion simplification*

        check_champion_reducibility
            Whether, after QMLA has determined a champion model, to test that 
            champion model - i.e. whether any of its parameters are negligible 
            and can be omitted.
        learned_param_limit_for_negligibility
            Threshold to consider a learned parmaeter negligible in the 
            champion reduction test.
        reduce_champ_bayes_factor_threshold
            The Bayes factor by which a proposed reduced champion 
            (omitting negligible parameters)
            must defeat the nominated champion, in order to be declared 
            champion in its place. 

        *Infrastructure* (shouldn't need to be replaced by custom GR).

        storage
            Generic storage unit which is exported to the QMLA instance and stored. 
            This allows for access to GR-specific data after QMLA has finished, 
            for analysis and plotting on a cross-instance basis.         
        spawn_stage
            list which is used by GR to determine what is the current stage of development.
            This can be checked against in :meth:`~qmla.growth_rules.GrowthRule.check_tree_completed`.
            e.g. calls to :meth:`~qmla.growth_rules.GrowthRule.generate_models` can append the list with 
            an indicator 'stage_1_complete'. 
            Subsequent calls can check ``if self.spawn_stage[-1] == 'stage_1_complete ... ``
            to design models according to the current stage. 
            By default, GR terminates model generation stage `` if self.spawn_stage[-1] == 'Complete'
        spawn_step
            Number of times spawn method (:meth:`~qmla.growth_rules.GrowthRule.generate_models`)
            has been called. 
            By default, GR terminates model generation stage `` if self.spawn_step == self.max_spawn_depth``. 
        prune_step
            Number of times prune method (:meth:`~qmla.growth_rules.GrowthRule.tree_pruning`)
            has been called.
            By default, when spawning and pruning are both completed, GR nominates a champion, 
            and is then terminated. 

        * Miscellaneous * 
        
        track_cov_mtx
            Whether to store the covariance matrix at the end of every experiment. 
            In general it is unnecessary to store. 
        

        """

        # Tree development
        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.tree_completed_initially = False
        self.prune_completed_initially = False
        self.max_spawn_depth = 10
        self.max_num_qubits = 5
        self.max_num_probe_qubits = 6
        self.num_top_models_to_build_on = 1

        # Rating models
        self.ratings_class = qmla.growth_rules.rating_system.ModifiedEloRating(
            initial_rating=1000,
            k_const=30
        ) 

        # Comparisons within branches
        self.branch_champion_selection_stratgey = 'number_comparison_wins' 
        self.branch_comparison_strategy = 'all'
        
        # Champion simplification
        self.check_champion_reducibility = True
        self.learned_param_limit_for_negligibility = 0.05
        self.reduce_champ_bayes_factor_threshold = 1e1

        # Infrastructure for tracking
        self.storage = qmla.utilities.StorageUnit()
        # self.model_branches = {}
        self.spawn_stage = [None]
        self.spawn_step = 0
        self.prune_complete = False
        self.prune_step = 0 
        self.track_cov_mtx = False # only sometimes want on for some plots


    def _setup_logistics(self):
        r"""
        Logistics for timing request etc.

        *Timing*

        On a compute cluster, we run QMLA by submitting jobs. 
        Those jobs need to specify the number of cores to request, 
        and the time required. These are computed in the 
        ``time_reuqired_calculation`` script, using the details specified here. 

        max_num_models_by_shape
            How many models to allow per number of qubits
        num_processes_to_parallelise_over
            Number of cores to request.
            On a single node, there are 16 cores available
            In general we use 1 core as the master which runs the
            :class:`~qmla.QuantumModelLearningAgent` instance, 
            and N worker processes, so :math:`N \leq 15` to fit on a 
            single node. 
            It is advisable to request as few processes (i.e. cores)
            as your growth rule needs, as the job scheduler favours less 
            resource-expensive jobs, so your jobs will spend less time in 
            the queue on the compute cluster.
        timing_insurance_factor
            A blunt tool to correct the estimate of time required 
            as determined by the ``time_request_calculation``. 
            Can be any float. 
            Should be set :math:`\neq 1` when you find the jobs are requesting 
            far too little time and are not finishing, or requesting too much
            which places them on slower queues. 

        """

        self.max_num_models_by_shape = {
            1: 0,
            2: 1,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 6
        self.timing_insurance_factor = 1



    ##########
    # Section: System (true model) infomation
    ##########

    def true_model_latex(self):
        r""" Latex representation of true model."""
        return self.latex_name(self.true_model)

    @property
    def true_model_terms(self):
        r""" Terms (as latex strings) which make up the true model"""
        true_terms = construct_models.get_constituent_names_from_name(
            self.true_model
        )

        latex_true_terms = [
            self.latex_name(term) for term in true_terms
        ]

        self.true_op_terms = set(sorted(latex_true_terms))

        return self.true_op_terms

    def get_true_parameters(
        self,
    ):  
        r"""
        Retrieve parameters of the true model and use them to construct the true Hamiltonian. 

        True parameters are set once per run and shared by all instances within that run. 
        Therefore the true parameters are generated only once 
        by :meth:`~qmla.set_shared_parameters`, and stored to a file which
        is accessible by all instances within the run. 

        This method retrieves those shared true parameters and stores them for use by the 
        :class:`~qmla.QuantumModelLearningAgent` instance and its subsidiary models and methods. 
        It then uses the true parameters to construct ``true_hamiltonian`` for the GR.   

        """      

        # get true data from pickled file
        try:
            true_config = pickle.load(
                open(
                    self.true_params_path, 
                    'rb'
                )
            )
            self.true_params_list = true_config['params_list']
            self.true_params_dict = true_config['params_dict']
        except:
            self.true_params_list = []
            self.true_params_dict = {}

        true_ham = None
        for k in list(self.true_params_dict.keys()):
            param = self.true_params_dict[k]
            mtx = construct_models.compute(k)
            if true_ham is not None:
                true_ham += param * mtx
            else:
                true_ham = param * mtx
        self.true_hamiltonian = true_ham

    def get_measurements_by_time(
        self
    ):
        r"""
        Measure the true model for a series of times. 

        In some experiment design heuristics, 
        those prescribed times are the only ones available to 
        the learning procedure. 
        Other heuristics allow the choice of any experimental time 
        in principle. 
        In either case, the measurements generated here are computed using the 
        ``plot_probes``, which are shared by all QMLA instances within the run. 
        They are used for all dynamics plots. 
        """

        try:
            true_info = pickle.load(
                open(
                    self.true_params_path, 'rb'
                )
            )
        except:
            print("Failed to load true params from path", self.true_params_path)
            raise

        self.true_params_dict = true_info['params_dict']
        true_ham = None
        for k in list(self.true_params_dict.keys()):
            param = self.true_params_dict[k]
            mtx = construct_models.compute(k)
            if true_ham is not None:
                true_ham += param * mtx
            else:
                true_ham = param * mtx
        self.true_hamiltonian = true_ham

        true_ham_dim = construct_models.get_num_qubits(self.true_model)
        plot_probes = pickle.load(
            open(
                self.plot_probes_path, 
                'rb'
            )
        )
        probe = plot_probes[true_ham_dim]

        plot_lower_time = 0
        plot_upper_time = self.max_time_to_consider
        if self.plot_time_increment is not None:
            raw_times = list(np.arange(
                plot_lower_time, 
                plot_upper_time, 
                self.plot_time_increment
            ))
            self.log_print([
                "Getting plot times from plot time increment. Raw times:", raw_times,
                "lower={}; upper={}; incr={}".format(plot_lower_time, plot_upper_time, self.plot_time_increment)
            ])
        else:
            num_datapoints_to_plot = 300
            raw_times = list(np.linspace(
                0,
                plot_upper_time,
                num_datapoints_to_plot + 1
            ))

        plot_times = [np.round(a, 2) if a>0.1 else a for a in raw_times]
        plot_times = sorted(plot_times)
        self.log_print([
            "plot times:", plot_times,
        ])

        self.measurements = {
            t : self.expectation_value(
                ham = self.true_hamiltonian, 
                t = t, 
                state = probe
            )
            for t in plot_times
        }
        return self.measurements


    ##########
    # Section: Functionality wrappers
    ##########

    # Measurement
    def expectation_value(
        self,
        **kwargs
    ):
        r"""
        Call the GR's ``expectation_value_function`` to compute quantum likelihood.

        Compute the probability of measuring in some basis, to be used as likelihood. 
        The default probability is that of the expectation value. 
        Given an input state :math:`\| \psi \rangle`, 
        :math:`P(\hat{H}, t, \| \psi \rangle) 
        = \| \langle \| e^{-i \hat{H} t} \| \psi \rangle \|^2`.
        However it is possible to use alternative measurements, 
        for instance corresponding to a physical measurement scheme such 
        as Hahn echo or Ramsey sequences. 


        Modular functions here must take as parameters

            ham 
                Hamiltonian to compute probability of
            t
                time to evolve ``ham`` for
            state
                proobe state to compute probability with 
            **kwargs
                any further inputs required can be passed as kwargs

        Modular functions must return 
            :math:`P` : the probability of measurement according to 
            custom requirements, to be used as likelihood in 
            `(interactive) quantum likelihood estimation`. 

        """ 

        return self.expectation_value_function(
            **kwargs
        )

    # Probe states
    def generate_probes(
        self,
        probe_maximum_number_qubits=None, 
        store_probes=True,
        **kwargs
    ):
        r""" 
        Call the GR's probe generation methods to set the system and simulator probes. 

        In general it is possible for the system and simulator to 
        have different probe states (e.g. due to noise).
        These can be generated from the same or different methods. 
        if ``shared_probes is True``, then ``probe_generation_function`` 
        is called once and the same probes are used for the system as simulator. 
        else ``simulator_probe_generation_function`` is called for the simulator
        probes. 

        Probe generation methods must take parameters
        
            max_num_qubits
                number of qubits to go up to when generating probes
            num_probes
                number of probces to produce

        Probe generation methods must return
            probe_dict
                A set of probes with ``num_probes`` states for each of 
                1, ..., N qubits up to ``max_num_qubits``. 
                Probe dictionaries should have keys which are tuples of the
                number of qubits and a probe ID, i.e. 
                ``(probe_id, num_qubits)``.

        :param int probe_maximum_number_qubits: 
            how many qubits to compose probes up to. 
            Can be left None, in which case assigned based on GR's 
            ``max_num_qubits``, or forced to a different value by passing 
            to function call. 
        :param bool store_probes: whether to assign the generated probes 
            to the GR instance. 
            If False, probe dict is just returned .
        :returns dict new_probes: (if not storing)
            dictionary of probes returned from probe generation
            function, fulfilling the requirements outlined above. 
        """

        if probe_maximum_number_qubits is None: 
            probe_maximum_number_qubits = self.max_num_probe_qubits
        self.log_print([
            "System Generate Probes called",
            "probe max num qubits:", probe_maximum_number_qubits
        ])

        # Generate a set of probes
        new_probes = self.probe_generation_function(
            max_num_qubits=probe_maximum_number_qubits,
            num_probes=self.num_probes,
            **kwargs
        )

        # Store or return the generated probes
        if store_probes:
            self.probes_system = new_probes
            if self.shared_probes == True:
                # Assign probes for simulator 
                self.probes_simulator = self.probes_system
            else:
                self.probes_simulator = self.simulator_probe_generation_function(
                    max_num_qubits=probe_maximum_number_qubits,
                    num_probes=self.num_probes,
                    **kwargs
                )
        else:
            return new_probes

    def plot_probe_generator(
        self,
        probe_maximum_number_qubits=None, 
        **kwargs
    ):
        r"""
        Call the GR's ``plot_probe_generation_function``. 

        Generates a set of probes against which to compute measurements for plotting purposes. 
        The same probe dict is used by all QMLA instances within a run for consistency. 

        Plot probe generation methods must adhere to the same rules 
        as in :meth:`~qmla.growth_rules.GrowthRule.generate_probes`. 
        
        :param int probe_maximum_number_qubits: 
            how many qubits to compose probes up to. 
            Can be left None, in which case assigned based on GR's 
            ``max_num_qubits``, or forced to a different value by passing 
            to function call. 
        :return dict plot_probe_dict: 
            set of states against which all models are plotted
            over time in dynamics plots.
        """

        if probe_maximum_number_qubits is None: 
            probe_maximum_number_qubits = self.max_num_probe_qubits

        # Generate probes
        plot_probe_dict =  self.plot_probe_generation_function(
            max_num_qubits=probe_maximum_number_qubits,
            num_probes=1,
            **kwargs
        )

        # Replace tuple like key returned, with just dimension.
        for k in list(plot_probe_dict.keys()):
            plot_probe_dict[k[1]] = plot_probe_dict.pop(k)
        
        # Store the probes 
        self.plot_probe_dict = plot_probe_dict
        
        return plot_probe_dict


    # Experiment design
    def heuristic(
        self,
        **kwargs
    ):
        r"""
        Call the GR's ``model_heuristic_function`` to build an experiment design heuristic class. 

        The heuristic class is called upon to design experiments to perform on 
        the system during model learning. 
        
        Heuristics should inherit from :class:`~qmla.shared_functionality.BaseHeuristic`. 
        Details of requirements for custom heuristics can be found in the defintion of 
        :class:`~qmla.shared_functionality.BaseHeuristic`.
        # TODO clear up - the heuristic is a class, not a function
        """

        return self.model_heuristic_function(
            **kwargs
        )

    # QInfer interface
    def qinfer_model(
        self, 
        **kwargs
    ):
        r"""
        Call the GR's ``qinfer_model_class`` to build the interface with QInfer used for model learning. 

        The default QInfer model class, and details of what to include in custom 
        classes, can be found in :class:`~qmla.shared_functionality.QInferModelQMLA`. 
        
        """

        return self.qinfer_model_class(
            **kwargs
        )

    # Prior parameterisation distribution
    def get_prior(
        self,
        model_name,
        **kwargs
    ):
        r"""
        Call the GR's ``prior_distribution_generator`` function. 

        :param str model_name: 
            model for which to construct a prior distribution
        :return qinfer.Distribution prior: 
            N-dimensional distribution used by QInfer as the starting
            distribution for learning model parameters. 
        """

        self.prior = self.prior_distribution_generator(
            model_name=model_name,
            prior_specific_terms=self.gaussian_prior_means_and_widths,
            param_minimum=self.min_param,
            param_maximum=self.max_param,
            random_mean=self.prior_random_mean,
            **kwargs
        )
        return self.prior

    # Map model name strings to latex representation
    def latex_name(
        self,
        name,
        **kwargs
    ):
        r"""
        Call the GR's ``latex_model_naming_function``. 

        Map a model name (string) to its LaTeX representation. 

        :param str name: name of model to map.
        :return str latex_name: representation of input model as LaTeX string. 
        """

        latex_name = self.latex_model_naming_function(name, **kwargs)
        return latex_name

    # Assign branch to model for visual representation of GR as tree
    def name_branch_map(
        self,
        latex_mapping_file,
        **kwargs
    ):
        
        r"""
        Assign branch to model for visual representation of GR as tree.

        Only used for attempt to plot the QMLA instance as a single tree, 
        which is often not suitable, so this is not essential. 
        
        """
        import qmla.shared_functionality.branch_mapping
        return qmla.shared_functionality.branch_mapping.branch_computed_from_qubit_and_param_count(
            latex_mapping_file=latex_mapping_file,
            **kwargs
        )

    ##########
    # Section: Tree growth
    ##########

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        r"""
        Determine the next set of models for this growth rule. 
        """
        # default is to just return given model list and set spawn stage to
        # complete
        return model_list


    def tree_pruning(
        self,
        previous_prune_branch,
    ):
        r"""
        Get next model set through pruning. 
        """
        self.prune_step += 1
        prune_step = self.prune_step
        pruning_models = []
        pruning_sets = []
        self.log_print([
            "Pruning within {}".format(self.growth_generation_rule),
            "Branches:", self.tree.branches
        ])
        if prune_step == 1:
            child_parent_pairs = []
            for branch in self.tree.branches.values():
                pruning_models.append(branch.champion_name)
                self.log_print([
                    "Getting child/parents for branch", branch.branch_id
                ])
                try:
                    champ = branch.champion_id
                    parent_champ = branch.parent_branch.champion_id
                    pair = (champ, parent_champ)
                    if champ != parent_champ:
                        pruning_sets.append(pair)
                except:
                    self.log_print([
                        "Branch has no parent:", branch.branch_id
                    ])
                    pass
    
        elif prune_step == 2:
            pruned_branch = self.tree.branches[previous_prune_branch]
            # check bayes factor compairsons on those from previous prune branch, 
            # which corresponds to parent/child collapse
            prune_collapse_threshold = 1e2 # TODO set as GR attribute
            prev_branch_models = []
            for l in list(zip(*pruned_branch.pairs_to_compare)):
                prev_branch_models.extend(list(l))
            prev_branch_models = list(set(prev_branch_models))

            models_to_prune = []
            for pair in pruned_branch.pairs_to_compare:
                id_1 = pair[0]
                id_2 = pair[1]
                mod_1 = pruned_branch.model_for_learning[id_1]
                try:
                    bf_1_v_2 = mod_1.model_bayes_factors[ float(id_2) ][-1]
                except:
                    self.log_print([
                        "couldnt find bf {}/{}. mod_{} BF:".format( 
                            id_1, 
                            id_2, 
                            id_1,
                            mod_1.model_bayes_factors
                        )
                    ])
                if bf_1_v_2 > prune_collapse_threshold:
                    models_to_prune.append(id_2)
                elif bf_1_v_2 < float(1 / prune_collapse_threshold):
                    models_to_prune(id_1)

                models_to_keep = list(
                    set(prev_branch_models)
                    - set(models_to_prune)
                )
                pruning_models = [
                    pruned_branch.models_by_id[m]
                    for m in models_to_keep
                ]
                pruning_sets = list(itertools.combinations(
                    models_to_keep, 
                    2
                ))
            self.prune_complete = True

        self.log_print([
            "Prune step {}. pruning models: {} \n pruning sets: {}".format(
                prune_step, 
                pruning_models, 
                pruning_sets
            )
        ])
        if len(pruning_models) == 1:
            self.prune_complete = True
        self.log_print(["Returning from pruning fnc"])
        return list(set(pruning_models)), pruning_sets
    
    def check_tree_pruned(self, prune_step, **kwargs):
        if prune_step >= 2 : 
            return True
        else:
            return False

    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if self.tree_completed_initially:
            return True
        elif spawn_step >= self.max_spawn_depth:
            return True
        else:
            return False


    def nominate_champions(self):
        final_branch = self.tree.branches[ max(self.tree.branches.keys()) ]
        self.log_print([
            "Nominating champion as champion of final branch {}: {}".format(
                final_branch.branch_id, 
                final_branch.champion_name
            )
        ])
        return [final_branch.champion_name]

    ##########
    # Section: Wrap up
    ##########

    def finalise_model_learning(self, **kwargs):
        self.log_print([" GR {} finished.".format(self.growth_generation_rule)])


    def growth_rule_finalise(self):
        # TODO consolidate this method with finalise_model_learning()
        # do whatever is needed to wrap up growth rule
        # e.g. store data required for analysis
        pass
        
    def growth_rule_specific_plots(
        self,
        save_directory, 
        **kwargs
    ):
        self.log_print(
            ['No growth rule plots specified.']
        )

    ##########
    # Section: Utilities
    ##########

    def log_print(
        self,
        to_print_list
    ):
        identifier = "[Growth: {}]".format(self.growth_generation_rule)
        if type(to_print_list) != list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(
                identifier,
                str(to_print),
                file=write_log_file,
                flush=True
            )

    def store_growth_rule_configuration(
        self, 
        path_to_pickle_config = None,
        **kwargs
    ):
        dict_for_storage = self.__dict__
        if path_to_pickle_config is not None: 
            pickle.dump(
                dict_for_storage,
                open(
                    path_to_pickle_config, 'wb'
                )                
            )
        return dict_for_storage

    def overwrite_growth_class_methods(
        self,
        **kwargs
    ):
        # print("[GrowthRule] overwrite_growth_class_methods. kwargs", kwargs)
        kw = list(kwargs.keys())

        attributes = [
            'probe_generator'
        ]

        for att in attributes:

            if att in kw and kwargs[att] is not None:
                print("Resetting {} to {}".format(att, kwargs[att]))
                self.__setattr__(att, kwargs[att])
