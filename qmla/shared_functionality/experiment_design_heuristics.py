import qinfer as qi
import numpy as np
import scipy as sp
import random 
import math
import copy 
import itertools

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inspect import currentframe, getframeinfo
try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

import qmla.logging


frameinfo = getframeinfo(currentframe())

__all__ = [
    'ExperimentDesignHueristic', 
    'MultiParticleGuessHeuristic',
    'MixedMultiParticleLinspaceHeuristic',
    'VolumeAdaptiveParticleGuessHeuristic'
]

def identity(arg): return arg

class ExperimentDesignHueristic(qi.Heuristic):
    """
    Experiment Design Heuristic base class, to be inherited by specific implementations. 
    This object has access to the QInfer Updater and Model objects, so it can, e.g., 
    sample from the particle distribution, to use these values in the design of a new experiment. 

    :param updater: QInfer updater for SMC 
    :type updater: QInfer Updater object
    :param model_id: ID of model under study, defaults to 1
    :type model_id: int
    :param oplist: list of matrices representing the operators constituting this model, defaults to None
    :type oplist: list, optional
    :param norm: type of norm to use, defaults to 'Frobenius'
    :type norm: str, optional
    :param inv_field: inversion field to use (legacy - should not matter) defaults to 'x_'
    :type inv_field: str, optional
    :param t_field: name of field corresponding to $t$, defaults to 't'
    :type t_field: str, optional
    :param maxiters: manimum number of iterations to attempt to find distinct particles from the distribution, defaults to 10
    :type maxiters: int, optional
    :param other_fields: optional further fields, defaults to None
    :type other_fields: list, optional
    :param inv_func: inverse function, used by QInfer, (legacy - should not matter) defaults to identity
    :type inv_func: function, optional
    :param t_func: function for computing $t$, defaults to identity
    :type t_func: function, optional
    :param log_file: path to log file, defaults to 'qmla_log.log'
    :type log_file: str, optional
    """    

    def __init__(
        self,
        updater,
        model_id=1, 
        oplist=None,
        norm='Frobenius',
        inv_field='x_',
        t_field='t',
        maxiters=10,
        other_fields=None,
        inv_func=identity,
        t_func=identity,    
        log_file='qmla_log.log',    
        **kwargs
    ):
        super().__init__(updater)
        # Most importantly - access to updater and underlying model
        self._model_id = model_id
        self._updater = updater
        self._model = updater.model

        # Other useful attributes passed
        self._norm = norm
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._other_fields = other_fields if other_fields is not None else {}
        self._maxiters = maxiters
        self._oplist = oplist
        self._num_experiments = kwargs['num_experiments']
        self._figure_format = kwargs["figure_format"]
        self._log_file = log_file

        # probe ID
        self.probe_id = 0
        self.probe_rotation_frequency = 5
        self.num_probes = kwargs['num_probes']

        # storage infrastructure
        self.heuristic_data = {} # to be stored by model instance
        self._resample_epochs = []
        self._volumes = []
        self.effective_sample_size = []
        self._times_suggested = []
        self._label_fontsize = 10 # consistency when plotting

    def _get_exp_params_array(self, epoch_id):
        r"""Return an empty array with a position for every experiment design parameter."""
        experiment_params = np.empty(
            (1,),
            dtype=self._model.expparams_dtype
        )

        # fill in particle in expparams
        particle = self._updater.sample()
        n_params = particle.shape[1]

        for i in range(n_params):
            p = particle[0][i]            
            corresponding_expparam = self._model.modelparam_names[i]
            experiment_params[corresponding_expparam] = p

        # choose probe id
        if epoch_id % self.probe_rotation_frequency == 0 :
            self.probe_id += 1
            if self.probe_id >= self.num_probes: 
                self.probe_id = 0 
        experiment_params['probe_id'] = self.probe_id

        return experiment_params

    def log_print(
        self,
        to_print_list
    ):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self._log_file,
            log_identifier="Heuristic {}".format(self._model_id) # TODO add heuristic name 
        )

    def __call__(self, **kwargs):
        """By calling the heuristic, it produces an experiment to be performed to learn upon.

        :return: all necessary data to perform an experiment, e.g. evolution time and probe ID.  
        :rtype: named tuple
        """        
        # Process some data from the model first
        try:
            current_volume = kwargs['current_volume']
        except:
            current_volume = None
        self._volumes.append(current_volume)

        if self._updater.just_resampled:
            self._resample_epochs.append( kwargs['epoch_id'] -1 )
        
        self.effective_sample_size.append(self._updater.n_ess)

        # Design a new experiment
        new_experiment =  self.design_experiment(**kwargs)
        new_time = new_experiment['t']
        if new_time > 1e6: 
            # TODO understand cutoff at which time 
            # calculation becomes unstable
            new_time = np.random.uniform(1e5, 1e6)
            # self.log_print([
            #     "Time too high -> randomising to ", new_time
            # ])

        if 'force_time_choice' in kwargs:
            new_time = kwargs['force_time_choice']
        self._times_suggested.append(new_time)
        new_experiment['t'] = new_time
        return new_experiment

    def design_experiment(self, **kwargs):
        r""" 
        Design an experiment. 
        Children classes can overwrite this function to implement custom logic 
            for the deisggn of experiments. 

        """
        raise RuntimeError(
            "experiment design method not written for this heuristic."
        )
    
    def finalise_heuristic(self, **kwargs):
        r""" Any functionality the user wishes to happen at the final call to the heuristic."""
        self.log_print([
            "{} Resample epochs: {}".format(
                len(self._resample_epochs), 
                self._resample_epochs,
            )
            # "\nTimes suggested:", self._times_suggested
        ])

    def plot_heuristic_attributes(
        self, 
        save_to_file, 
        **kwargs
    ):
        """  
        Summarise the heuristic used for the model training through several plots. 

            volume of distribution at each experiment
            
            time designed by heuristic for each experiment

            effecitve sample size at each experiment, used to determine when to resample 

        :param save_to_file: path to which the summary figure is stored
        :type save_to_file: path
        """        

        plots_to_include = [
            'volume', 'times_used', 'effective_sample_size'
        ]
        
        plt.clf()
        nrows = len(plots_to_include)
        lf = LatexFigure(
            gridspec_layout=(nrows, 1)
        )

        if 'volume' in plots_to_include:
            ax = lf.new_axis()
            self._plot_volumes(ax = ax)
            ax.legend()

        if 'times_used' in plots_to_include:
            ax = lf.new_axis()
            self._plot_suggested_times(ax = ax)
            ax.legend()

        if 'effective_sample_size' in plots_to_include:
            ax = lf.new_axis()
            self._plot_effective_sample_size(ax = ax)
            ax.legend()

        # Save figure
        self.log_print(["LatexFigure has size:", lf.size])
        lf.save(save_to_file, file_format=self._figure_format)


    def _plot_suggested_times(self, ax, **kwargs):
        full_epoch_list = range(len(self._times_suggested))
        ax.scatter(
            full_epoch_list, 
            self._times_suggested,
            label = r"$t \sim k \ \frac{1}{V}$",
            s = 5,
        )
        ax.set_title('Experiment times', fontsize=self._label_fontsize)
        ax.set_ylabel('Time', fontsize=self._label_fontsize)
        ax.set_xlabel('Epoch', fontsize=self._label_fontsize)
        self._add_resample_epochs_to_ax(ax = ax)
        ax.semilogy()

    def _plot_volumes(self, ax, **kwargs):
        full_epoch_list = range(len(self._volumes))
        ax.plot(
            full_epoch_list, 
            self._volumes,
            label = 'Volume',
        )
        ax.set_title('Volume', fontsize=self._label_fontsize)
        ax.set_ylabel('Volume', fontsize=self._label_fontsize)
        ax.set_xlabel('Epoch', fontsize=self._label_fontsize)
        self._add_resample_epochs_to_ax(ax = ax)
        ax.semilogy()

    def _plot_effective_sample_size(self, ax, **kwargs):
        full_epoch_list = range(len(self.effective_sample_size))
        ax.plot(
            full_epoch_list, 
            self.effective_sample_size,
            label = r"$N_{ESS}$",
        )

        resample_thresh = self._updater.resample_thresh
        ax.axhline(
            resample_thresh * self.effective_sample_size[0], 
            label="Resample threshold ({}%)".format(resample_thresh*100),
            color = 'grey',
            ls = '-',
            alpha = 0.5
        )
        if resample_thresh != 0.5:
            ax.axhline(
                self.effective_sample_size[0] / 2, 
                label="50%",
                color = 'grey',
                ls = '--', 
                alpha = 0.5
            )

        ax.set_title('Effective Sample Size', fontsize=self._label_fontsize)
        ax.set_ylabel('$N_{ESS}$', fontsize=self._label_fontsize)
        ax.set_xlabel('Epoch', fontsize=self._label_fontsize)
        self._add_resample_epochs_to_ax(ax = ax)
        ax.legend()
        ax.set_ylim(0, self.effective_sample_size[0]*1.1)
        # ax.semilogy()


    def _add_resample_epochs_to_ax(self, ax, **kwargs):
        c = 'grey'
        a = 0.5
        ls = ':'
        if len(self._resample_epochs) > 0:
            ax.axvline(
                self._resample_epochs[0], 
                ls=ls, 
                color=c, 
                label="Resample", 
                alpha=a
            )

            for e in self._resample_epochs[1:]:
                ax.axvline(
                    e, ls=ls, color=c, alpha=a
                )


class MultiParticleGuessHeuristic(ExperimentDesignHueristic):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.log_print(["Particle Guess Heuristic"])

    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        idx_iter = 0
        while idx_iter < self._maxiters:
            sample = self._updater.sample(n=2) 
            x, xp = sample[:, np.newaxis, :]
            if self._model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1

        if self._model.distance(x, xp) == 0:
            self.log_print([
                "x,xp={},{}".format(x,xp)
            ])
            raise RuntimeError(
                "PGH did not find distinct particles in \
                {} iterations.".format(self._maxiters)
            )

        d = self._model.distance(x, xp)
        new_time = 1 / d

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        eps['t'] = new_time

        # get sample from x
        particle = self._updater.sample()
        # self.log_print(["Particle for IQLE=", particle])
        n_params = particle.shape[1]

        for i in range(n_params):
            p = particle[0][i]            
            corresponding_expparam = self._model.modelparam_names[i]
            eps[corresponding_expparam] = p

        return eps

class MixedMultiParticleLinspaceHeuristic(ExperimentDesignHueristic):
    r"""
    First half of experiments are standard MPGH, then force times evenly spaced 
    between 0 and max_time.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.log_print(["Mixed Particle Guess Heuristic"])

        self.max_time_to_enforce = kwargs['max_time_to_enforce']
        self.count_number_high_times_suggested = 0 

        self.num_epochs_for_first_phase = self._num_experiments / 2
        # generate a list of times of length Ne/2
        # evenly spaced between 0, max_time (from exploration_strategy)
        # then every t in that list is learned upon once. 
        # Higher Ne means finer granularity 
        # times are leared in a random order (from random.shuffle below)
        num_epochs_to_space_time_list = math.ceil(
            self._num_experiments - self.num_epochs_for_first_phase
        )
        t_list = list(np.linspace(
            0, 
            self.max_time_to_enforce,
            num_epochs_to_space_time_list + 1
        ))
        t_list.remove(0)  # dont want to waste an epoch on t=0
        t_list = [np.round(t, 2) for t in t_list]
        # random.shuffle(t_list)
        self._time_list = itertools.cycle( t_list )
        

    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        idx_iter = 0
        while idx_iter < self._maxiters:

            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1

        if self._model.distance(x, xp) == 0:
            raise RuntimeError(
                "PGH did not find distinct particles in \
                {} iterations.".format(self._maxiters)
            )

        eps = self._get_exp_params_array(epoch_id = epoch_id)

        if epoch_id < self.num_epochs_for_first_phase:
            d = self._model.distance(x, xp)
            new_time = self._t_func(
                1 / d
            )
        else:
            new_time = next(self._time_list)
        
        if new_time > self.max_time_to_enforce:
            self.count_number_high_times_suggested += 1
        
        if epoch_id == self._num_experiments - 1 : 
            self.log_print([
                "Number of suggested t > t_max:", self.count_number_high_times_suggested 
            ])
        eps['t'] = new_time
        return eps


class SampleOrderMagnitude(ExperimentDesignHueristic):
    
    def __init__(
        self,
        updater,
        **kwargs
    ):
        super().__init__(updater, **kwargs)
        self.count_order_of_magnitudes =  {}
        self.force_order = None
    
    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        experiment = self._get_exp_params_array(epoch_id = epoch_id) # empty experiment array
        
        # sample from updater
        idx_iter = 0
        while idx_iter < self._maxiters:

            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1

        cov_mtx = self._updater.est_covariance_mtx()
        orders_of_magnitude = np.log10(
            np.sqrt(np.abs(np.diag(cov_mtx)))
        ) # of the uncertainty on the individual parameters

        orders_of_magnitude[orders_of_magnitude < 1] = 1 # lower bound

        probs_of_orders = [
            i/sum(orders_of_magnitude) for i in orders_of_magnitude
        ] # weight of each order of magnitude
        # sample from the present orders of magnitude
        selected_order = np.random.choice(
            a = orders_of_magnitude, 
            p = probs_of_orders
        )
        if self.force_order is not None: 
            selected_order = self.force_order

        try:
            self.count_order_of_magnitudes[ np.round(selected_order)] += 1
        except:
            self.count_order_of_magnitudes[ np.round(selected_order)] = 1

        idx_params_of_similar_uncertainty = np.where(
            np.isclose(orders_of_magnitude, selected_order, atol=1)
        ) # within 1 order of magnitude of the max

        # change the scaling matrix used to calculate the distance
        # to place importance only on the sampled order of magnitude
        self._model._Q = np.zeros( len(orders_of_magnitude) )
        for idx in idx_params_of_similar_uncertainty:
            self._model._Q[idx] = 1

        d = self._model.distance(x, xp)
        new_time = 1 / d

        if self.force_order == 9:
            new_time *= 100
        experiment[self._t] = new_time

        # print("Available orders of magnitude:", orders_of_magnitude)
        # print("Selected order = ", selected_order)    
        # print("x= {}".format(x))
        # print("x'={}".format(xp))
        # print("Distance = ", d)
        # print("Distance order mag=", np.log10(d))
        # print("=> time=", new_time)
        
        return experiment

    def finalise_heuristic(self):
        super().finalise_heuristic()

        self.log_print([
            "count_order_of_magnitudes:", self.count_order_of_magnitudes
        ])


class SampledUncertaintyWithConvergenceThreshold(ExperimentDesignHueristic):
    
    def __init__(
        self,
        updater,
        **kwargs
    ):
        super().__init__(updater, **kwargs)
        
        self._qinfer_model = self._model
        cov_mtx = self._updater.est_covariance_mtx()
        self.initial_uncertainties = np.sqrt(np.abs(np.diag(cov_mtx)))
        self.track_param_uncertainties = np.zeros(self._qinfer_model.n_modelparams)
        self.selection_criteria = 'relative_volume_decrease' # 'hard_code_6' # 'hard_code_6_9_magnitudes'
        self.count_order_of_magnitudes =  {}
        self.all_count_order_of_magnitudes =  {}
        self.counter_productive_experiments = 0 
        self.call_counter = 0 
        self._num_experiments = kwargs['num_experiments']
        self._num_exp_to_switch_magnitude = self._num_experiments / 2
        print("Heuristic - num experiments = ", self._num_experiments)
        print("epoch to switch target at:", self._num_exp_to_switch_magnitude)
        
    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        self.call_counter += 1
        experiment = self._get_exp_params_array(epoch_id = epoch_id) # empty experiment array
        
        # sample from updater
        idx_iter = 0
        while idx_iter < self._maxiters:

            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1

        current_param_est = self._updater.est_mean()
        cov_mtx = self._updater.est_covariance_mtx()
        param_uncertainties = np.sqrt(np.abs(np.diag(cov_mtx))) # uncertainty of params individually
        orders_of_magnitude = np.log10(
            param_uncertainties
        ) # of the uncertainty on the individual parameters
        param_order_mag = np.log10(current_param_est)
        relative_order_magnitude = param_order_mag / max(param_order_mag)
        weighting_by_relative_order_magnitude = 10**relative_order_magnitude
        self.track_param_uncertainties = np.vstack( 
            (self.track_param_uncertainties, param_uncertainties) 
        )
        
        if self.selection_criteria.startswith('hard_code'):
            if self.selection_criteria== 'hard_code_6_9_magnitudes':
                if self.call_counter > self._num_exp_to_switch_magnitude:
                    order_to_target = 6
                else:
                    order_to_target = 9
            elif self.selection_criteria == 'hard_code_6':
                order_to_target = 6                
            locations = np.where(
                np.isclose(orders_of_magnitude, order_to_target, atol=1)
            )
            weights = np.zeros( len(orders_of_magnitude) )
            weights[locations] = 1
            probability_of_param = weights / sum(weights)

        elif self.selection_criteria == 'relative_volume_decrease':
            # probability of choosing  order of magnitude 
            # of each parameter based on the ratio
            # (change in volume)/(current estimate)
            # for that parameter
            print("Sampling by delta uncertainty/ estimate")
            change_in_uncertainty = np.diff( 
                self.track_param_uncertainties[-2:], # most recent two track-params
                axis = 0
            )[0]
            print("change in uncertainty=", change_in_uncertainty)
            if np.all( change_in_uncertainty < 0 ):
                # TODO better way to deal with all increasing uncertainties
                print("All parameter uncertainties increased")
                self.counter_productive_experiments += 1
                weights = 1 / np.abs(change_in_uncertainty)

            else:
                # disregard changes which INCREASE volume:
                change_in_uncertainty[ change_in_uncertainty < 0 ] = 0            
                # weight = ratio of how much that change has decreased the volume 
                # over the current best estimate of the parameter
                weights = change_in_uncertainty / current_param_est
            weights *= weighting_by_relative_order_magnitude # weight the likelihood of selecting a parameter by its order of magnitude
            probability_of_param = weights / sum(weights)

        elif self.selection_criteria == 'order_of_magniutde':
            # probability directly from order of magnitude
            print("Sampling by order magnitude")
            probability_of_param = np.array(orders_of_magnitude) / sum(orders_of_magnitude)
        else:
            # sample evenly
            print("Sampling evenly")
            probability_of_param = np.ones(self._qinfer_model.n_modelparams)
       

        # sample from the present orders of magnitude
        selected_order = np.random.choice(
            a = orders_of_magnitude, 
            p = probability_of_param
        )
        try:
            self.count_order_of_magnitudes[ np.round(selected_order)] += 1
            self.all_count_order_of_magnitudes[np.round(selected_order)] += 1
        except:
            self.count_order_of_magnitudes[ np.round(selected_order)] = 1
            self.all_count_order_of_magnitudes[np.round(selected_order)] = 1

        idx_params_of_similar_uncertainty = np.where(
            np.isclose(orders_of_magnitude, selected_order, atol=1)
        ) # within 1 order of magnitude of the max

        self._model._Q = np.zeros( len(orders_of_magnitude) )
        for idx in idx_params_of_similar_uncertainty:
            self._qinfer_model._Q[idx] = 1

        d = self._qinfer_model.distance(x, xp)
        new_time = 1 / d
        experiment[self._t] = new_time

        print("Current param estimates:", current_param_est)
        try:
            print("Weights:", weights)
        except:
            pass

        print("probability_of_param: ", probability_of_param)
        print("orders_of_magnitude:", orders_of_magnitude)
        print("Selected order = ", selected_order)    
        print("x={}".format(x))
        print("xp={}".format(xp))
        print("Distance = ", d)
        print("Distance order mag=", np.log10(d))
        print("=> time=", new_time)
        
        return experiment

class VolumeAdaptiveParticleGuessHeuristic(ExperimentDesignHueristic):
    def __init__(
        self,
        updater,
        **kwargs
    ):
        super().__init__(updater, **kwargs)
        
        self.time_multiplicative_factor = 1
        self.derivative_frequency = self._num_experiments / 20
        self.burn_in_learning_time = 6 * self.derivative_frequency
        self.log_print([
            "Derivative freq:{} \t burn in:{}".format(
                self.derivative_frequency, self.burn_in_learning_time) 
        ])
        self.time_factor_boost = 10 # factor to increase/decrease by 
        self.derivatives = { 1:{}, 2:{} }
        self.time_factor_changes =  {'decreasing' : [], 'increasing' : [] }
        self.distances = []
        distance_metrics = [
            'cityblock', 'euclidean', 'chebyshev', 
            'canberra', 'braycurtis','minkowski', 
        ]
        self.designed_times = { m : {} for m in distance_metrics }
        self.distance_metric_to_use = 'euclidean'

    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):

        # Maybe increase multiplicative factor for time chosen later
        if (
            epoch_id % self.derivative_frequency == 0 
            and epoch_id > self.burn_in_learning_time
        ):
            current_volume = self._volumes[-1]
            previous_epoch_to_compare = int(-1 - self.derivative_frequency)
            try:
                previous_volume = self._volumes[ previous_epoch_to_compare ] 
            except:
                previous_volume = self._volumes[0] 
                self.log_print([
                    "Couldn't find {}th element of volumes: {}".format(
                        previous_epoch_to_compare, self._volumes)
                ])
            relative_change =  (1 - current_volume / previous_volume)

            self.log_print([
                "At epoch {} V_old/V_new={}/{}. relative change={}".format(
                    epoch_id, 
                    np.round(previous_volume, 2), np.round(current_volume, 2), 
                    relative_change)
            ])
            
            expected_change = 0.2 # N% decrease in volume after N experiments (?)
            if 0 < relative_change <= expected_change:
                self.time_multiplicative_factor *= self.time_factor_boost
                print("Epoch {} r={}. Increasing time factor: {}".format(epoch_id, relative_change, self.time_multiplicative_factor))
                self.time_factor_changes['increasing'].append(epoch_id)
            elif relative_change < -0.1 :
                self.time_multiplicative_factor /= self.time_factor_boost # volume increasing -> use lower times
                print("Epoch {} r={}. Decreasing time factor: {}".format(epoch_id, relative_change,self.time_multiplicative_factor))
                self.time_factor_changes['decreasing'].append(epoch_id)
            elif relative_change > 0.1:
                self.time_multiplicative_factor *= 1 # learning well enough
                print("Epoch {} r={}. Maintaining time factor: {}".format(epoch_id, relative_change, self.time_multiplicative_factor))

        # Select particles
        idx_iter = 0
        while idx_iter < self._maxiters:
            sample = self._updater.sample(n=2) 
            x, xp = sample[:, np.newaxis, :]
            if self._model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1

        if self._model.distance(x, xp) == 0:
            raise RuntimeError(
                "PGH did not find distinct particles in \
                {} iterations.".format(self._maxiters)
            )

        for method in self.designed_times:
            d = sp.spatial.distance.pdist(
                sample, 
                metric=method
            )
            self.designed_times[method][epoch_id] = 1/d

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        new_time = copy.copy(self.designed_times[self.distance_metric_to_use][epoch_id])
        new_time *= self.time_multiplicative_factor
        eps['t'] = new_time
        return eps

    def plot_heuristic_attributes(self, save_to_file, **kwargs):
        """Plot results related to the experiment design heuristic.
        Plots

            times generated by the heuristic;

            metric distances: distance between the two sampled particles under various measures;

            volume of the parameter distribution at each experiment.

        :param save_to_file: path to which to save the resultant figure.
        :type save_to_file: path
        """        
        
        plots_to_include = [
            'volume', 'metric_distances', 'times_used', #'derivatives',
        ]
        
        plt.clf()
        nrows = len(plots_to_include)
        lf = LatexFigure(
            gridspec_layout=(nrows, 1)
        )

        # Volume
        if 'volume' in plots_to_include:
            ax = lf.new_axis()
            self._plot_volumes(ax = ax)
            self.add_time_factor_change_points_to_ax(ax = ax)
            ax.legend()

        if 'times_used' in plots_to_include:
            ax = lf.new_axis()

            self._plot_suggested_times(ax = ax)
            self.add_time_factor_change_points_to_ax(ax = ax)
            ax.legend()

        if 'derivatives' in plots_to_include:
            # Volume Derivatives
            ax = lf.new_axis()
            ## first derivatives
            derivs = self.derivatives[1]
            epochs = sorted(derivs.keys())
            first_derivatives = [derivs[e] if e in derivs else None for e in epochs]
            ax.plot(
                epochs, first_derivatives, 
                label=r"$\frac{dV}{dE}$", 
                color='darkblue', marker='x'
            )

            ## second derivatives
            derivs = self.derivatives[2]
            epochs = sorted(derivs.keys())
            second_derivatives = [derivs[e] if e in derivs else None for e in epochs]
            ax.plot(
                epochs, second_derivatives, 
                label=r"$\frac{d^2V}{dE^2}$", 
                color='maroon', marker='+'
            )

            ax.axhline(0, ls='--', alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_yscale('symlog')
            ax.set_ylabel('Derivatives', fontsize=self._label_fontsize)
            ax.set_xlabel('Epoch', fontsize=self._label_fontsize)

            if len(self.time_factor_changes['decreasing']) > 0:
                ax.axvline(
                    self.time_factor_changes['decreasing'][0], 
                    ls='--', color='red', label='decrease k', alpha=0.5
                )

                for e in self.time_factor_changes['decreasing'][1:]:
                    ax.axvline(
                        e, ls='--', color='red', alpha=0.5
                    )
            if len(self.time_factor_changes['increasing']) > 0:
                ax.axvline(
                    self.time_factor_changes['increasing'][0], 
                    ls='--', color='green', label='increase k', alpha=0.5
                )

                for e in self.time_factor_changes['increasing'][1:]:
                    ax.axvline(
                        e, ls='--', color='green', alpha=0.5
                    )
            ax.legend(loc='lower right')

        if 'metric_distances' in plots_to_include:
            # Times by distance metrics
            ax = lf.new_axis()

            linestyles = itertools.cycle( 
                ['--', ':', '-.']
            )
            for method in self.designed_times:
                times_of_method = self.designed_times[method]
                epochs = sorted(times_of_method.keys())
                time_by_epoch = [times_of_method[e] for e in epochs]

                lb = method
                if self.distance_metric_to_use == method:
                    ls = '-'
                    lb += ' (used)'
                    alpha = 1
                else:
                    ls  = next(linestyles)
                    alpha = 0.75

                ax.plot(
                    epochs,
                    time_by_epoch,
                    label=lb,
                    ls=ls,
                    alpha=alpha
                )
            ax.legend(title='Distance metric') 
            ax.set_title('Raw time chosen by distance metrics', fontsize=self._label_fontsize)      
            ax.set_ylabel('Time', fontsize=self._label_fontsize)
            ax.set_xlabel('Epoch', fontsize=self._label_fontsize)
            ax.semilogy()

        # Save figure
        self.log_print(["LatexFigure has size:", lf.size])
        lf.save(save_to_file, file_format=self._figure_format)

    def add_time_factor_change_points_to_ax(self, ax):

        if len(self.time_factor_changes['decreasing']) > 0:
            ax.axvline(
                self.time_factor_changes['decreasing'][0], 
                ls='--', 
                color='red', 
                label=r"$k \rightarrow k / {}$".format(np.round(self.time_factor_boost, 2)), 
                alpha=0.5
            )

            for e in self.time_factor_changes['decreasing'][1:]:
                ax.axvline(
                    e, ls='--', color='red', alpha=0.5
                )
        if len(self.time_factor_changes['increasing']) > 0:
            ax.axvline(
                self.time_factor_changes['increasing'][0], 
                ls='--', 
                color='green', 
                # label='increase k', 
                label=r"$ k \rightarrow k \times {}$".format(np.round(self.time_factor_boost,2)), 
                alpha=0.5
            )

            for e in self.time_factor_changes['increasing'][1:]:
                ax.axvline(
                    e, ls='--', color='green', alpha=0.5
                )


class FixedNineEighthsToPowerK(ExperimentDesignHueristic):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._k = -25

    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        if epoch_id % 10 == 0:
            # don't want to increment at every single experiment when there are a lot to do. 
            # b/c t becomes far too big
            self._k += 1
        new_time = (9/8)**self._k

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        eps['t'] = new_time

        # get sample from x

        # particle = self._updater.sample()
        # n_params = particle.shape[1]

        # for i in range(n_params):
        #     p = particle[0][i]            
        #     corresponding_expparam = self._model.modelparam_names[i]
        #     eps[corresponding_expparam] = p

        return eps


class RandomTimeUpperBounded(ExperimentDesignHueristic):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_time_to_enforce = kwargs['max_time_to_enforce']

    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        new_time = random.uniform(0 , self.max_time_to_enforce)
        # new_time  = self._max_time

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        eps['t'] = new_time

        # get sample from x
        particle = self._updater.sample()
        n_params = particle.shape[1]

        for i in range(n_params):
            p = particle[0][i]            
            corresponding_expparam = self._model.modelparam_names[i]
            eps[corresponding_expparam] = p

        return eps


class FixedTimeTest(ExperimentDesignHueristic):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):
        new_time = 5
        # new_time  = self._max_time

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        eps['t'] = new_time

        # get sample from x
        particle = self._updater.sample()
        n_params = particle.shape[1]

        for i in range(n_params):
            p = particle[0][i]            
            corresponding_expparam = self._model.modelparam_names[i]
            eps[corresponding_expparam] = p

        return eps




class TimeList(ExperimentDesignHueristic):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.max_time_to_enforce = kwargs['max_time_to_enforce']
        min_t = self.max_time_to_enforce / self._num_experiments
        delta_t = 2*min_t # effectively how many iterations each time is eventually learned for
        time_list = np.arange(
            min_t, 
            self.max_time_to_enforce, 
            delta_t
        )
        self.log_print([
            "delta t for heuristic:", delta_t
        ])

        self.time_list = itertools.cycle(time_list)        


    def design_experiment(
        self,
        epoch_id=0,
        **kwargs
    ):

        new_time = next(self.time_list)

        eps = self._get_exp_params_array(epoch_id = epoch_id)
        eps['t'] = new_time

        return eps


