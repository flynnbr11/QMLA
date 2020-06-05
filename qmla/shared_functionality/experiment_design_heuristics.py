import qinfer as qi
import numpy as np
import scipy as sp
import random 
import math

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inspect import currentframe, getframeinfo

import qmla.logging


frameinfo = getframeinfo(currentframe())

__all__ = [
    'MultiParticleGuessHeuristic',
    'MixedMultiParticleLinspaceHeuristic',
    'VolumeAdaptiveParticleGuessHeuristic'
]

def identity(arg): return arg

def log_print(
    to_print_list, 
    log_file, 
):
    qmla.logging.print_to_log(
        to_print_list = to_print_list, 
        log_file = log_file, 
        log_identifier = 'Heuristic'
    )

class BaseHeuristicQMLA(qi.Heuristic):
    def __init__(
        self,
        updater,
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
        self._norm = norm
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._other_fields = other_fields if other_fields is not None else {}
        self._updater = updater
        self._model = updater.model
        self._maxiters = maxiters
        self._oplist = oplist
        self.num_experiments = kwargs['num_experiments']
        self.log_file = log_file
        
        self.times_suggested = []
        self.heuristic_data = {} # to be stored by model instance
        
    def _get_exp_params_array(self):
        r"""Return an empty array with a position for every experiment design parameter."""
        experiment_params = np.empty(
            (1,),
            dtype=self._model.expparams_dtype
        )
        return experiment_params

    def log_print(
        self,
        to_print_list
    ):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier="Heuristic" # TODO add heuristic name 
        )

    def __call__(self, **kwargs):
        raise RuntimeError(
            "__call__ method not written for this heuristic."
        )
    
    def summarise_heuristic(self, **kwargs):
        self.log_print([
            "Times suggested:", self.times_suggested
        ])

    def plot_heuristic_attributes(self, save_to_file, **kwargs):
        # Plot results related to heuristic here
        # and/or log_print some details. 
        pass


class MultiParticleGuessHeuristic(BaseHeuristicQMLA):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def __call__(
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
            raise RuntimeError(
                "PGH did not find distinct particles in \
                {} iterations.".format(self._maxiters)
            )

        d = self._model.distance(x, xp)
        eps = self._get_exp_params_array()
        new_time = 1 / d
        eps['t'] = new_time

        return eps


class MixedMultiParticleLinspaceHeuristic(BaseHeuristicQMLA):
    r"""
    First half of experiments are standard MPGH, then force times evenly spaced 
    between 0 and max_time.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_time_to_enforce = kwargs['max_time_to_enforce']
        self.count_number_high_times_suggested = 0 

        self.num_epochs_for_first_phase = self.num_experiments / 2
        # generate a list of times of length Ne/2
        # evenly spaced between 0, max_time (from growth_rule)
        # then every t in that list is learned upon once. 
        # Higher Ne means finer granularity 
        # times are leared in a random order (from random.shuffle below)
        num_epochs_to_space_time_list = math.ceil(
            self.num_experiments - self.num_epochs_for_first_phase
        )
        t_list = list(np.linspace(
            0, 
            self.max_time_to_enforce,
            num_epochs_to_space_time_list + 1
        ))
        t_list.remove(0)  # dont want to waste an epoch on t=0
        t_list = [np.round(t, 2) for t in t_list]
        # random.shuffle(t_list)
        self._time_list = iter( t_list )
        

    def __call__(
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

        eps = self._get_exp_params_array()

        if epoch_id < self.num_epochs_for_first_phase:
            d = self._model.distance(x, xp)
            new_time = self._t_func(
                1 / d
            )
        else:
            new_time = next(self._time_list)
        
        if new_time > self.max_time_to_enforce:
            self.count_number_high_times_suggested += 1
        
        if epoch_id == self.num_experiments - 1 : 
            log_print([
                "Number of suggested t > t_max:", self.count_number_high_times_suggested 
                ],log_file = self.log_file 
            )
        eps['t'] = new_time
        return eps


class SampleOrderMagnitude(BaseHeuristicQMLA):
    
    def __init__(
        self,
        updater,
        **kwargs
    ):
        super().__init__(updater, **kwargs)
        self.count_order_of_magnitudes =  {}
    
    def __call__(
        self,
        epoch_id=0,
        **kwargs
    ):
        experiment = self._get_exp_params_array() # empty experiment array
        
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

        probs_of_orders = [
            i/sum(orders_of_magnitude) for i in orders_of_magnitude
        ] # weight of each order of magnitude
        # sample from the present orders of magnitude
        selected_order = np.random.choice(
            a = orders_of_magnitude, 
            p = probs_of_orders
        )
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
        experiment[self._t] = new_time

        print("Available orders of magnitude:", orders_of_magnitude)
        print("Selected order = ", selected_order)    
        print("x= {}".format(x))
        print("x'={}".format(xp))
        print("Distance = ", d)
        print("Distance order mag=", np.log10(d))
        print("=> time=", new_time)
        
        return experiment


class SampledUncertaintyWithConvergenceThreshold(BaseHeuristicQMLA):
    
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
        
    def __call__(
        self,
        epoch_id=0,
        **kwargs
    ):
        self.call_counter += 1
        experiment = self._get_exp_params_array() # empty experiment array
        
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

class VolumeAdaptiveParticleGuessHeuristic(BaseHeuristicQMLA):
    def __init__(
        self,
        updater,
        **kwargs
    ):
        super().__init__(updater, **kwargs)
        
        self.volumes = []
        self.time_multiplicative_factor = 1
        self.derivative_frequency = self.num_experiments / 20
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

    def __call__(
        self,
        epoch_id=0,
        **kwargs
    ):
        current_params = kwargs['current_params'] 
        current_volume = kwargs['current_volume']
        if len(self.volumes) == 0: 
            self.volumes.append(current_volume)
            print("V0 = ", current_volume)
        self.volumes.append(current_volume)

        # Maybe increase multiplicative factor for time chosen later
        if (
            epoch_id % self.derivative_frequency == 0 
            and epoch_id > self.burn_in_learning_time
        ):
            try:
                previous_epoch_to_compare = int(-1 - self.derivative_frequency)
                previous_volume = self.volumes[ previous_epoch_to_compare ] 
            except:
                previous_volume = self.volumes[0] 
                self.log_print([
                    "Couldn't find {}th element of volumes: {}".format(
                        -1 - self.derivative_frequency, self.volumes)
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

        eps = np.zeros(
            (1,),
            dtype=self._model.expparams_dtype
        )

        new_time = self.designed_times[self.distance_metric_to_use][epoch_id]
        d = 1 / new_time
        new_time *= self.time_multiplicative_factor
        self.distances.append(d)
        eps['t'] = new_time
        return eps

    def plot_heuristic_attributes(self, save_to_file, **kwargs):

        plt.clf()
        label_fontsize = 20
        nrows = 3
        fig = plt.figure( 
            figsize=(15, 3*nrows)
        )

        gs = GridSpec(
            nrows = nrows,
            ncols = 1,
        )

        row = 0
        col = 0

        # Volume
        ax = fig.add_subplot(gs[row, 0])
        full_epoch_list = range(len(self.volumes))
        ax.plot(
            full_epoch_list, 
            self.volumes,
            label = 'Volume',
        )
        ax.set_title('Volume', fontsize=label_fontsize)
        ax.set_ylabel('Volume', fontsize=label_fontsize)
        ax.set_xlabel('Epoch', fontsize=label_fontsize)
        ax.semilogy()
        ax.legend()

        # Volume Derivatives
        row += 1
        ax = fig.add_subplot(gs[row, 0])
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
        ax.set_ylabel('Derivatives', fontsize=label_fontsize)
        ax.set_xlabel('Epoch', fontsize=label_fontsize)

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


        # Times by distance metrics
        row += 1
        ax = fig.add_subplot(gs[row, 0])

    
        for method in self.designed_times:
            times_of_method = self.designed_times[method]
            epochs = sorted(times_of_method.keys())
            time_by_epoch = [times_of_method[e] for e in epochs]

            if self.distance_metric_to_use == method:
                ls = '--'
            else:
                ls  = '-'
            ax.plot(
                epochs,
                time_by_epoch,
                label=method,
                ls=ls
            )
        ax.legend(title='Distance metric') 
        ax.set_title('Times chosen by distance metrics', fontsize=label_fontsize)      
        ax.set_ylabel('Time', fontsize=label_fontsize)
        ax.set_xlabel('Epoch', fontsize=label_fontsize)
        ax.semilogy()
        # ax.grid()

        # Save figure
        fig.tight_layout()
        fig.savefig(
            save_to_file
        )

