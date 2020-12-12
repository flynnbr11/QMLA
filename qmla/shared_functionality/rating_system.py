import sys
import os
import numpy as np
import random
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

import qmla.utilities

class RatingSystem():
    def __init__(
        self,
        initial_rating=1000,
        k_const=30, # constant in calculation of ELO rating (?),
        reset_rating_at_each_generation=True, 
    ):
        self.models = {}
        self.initial_rating = initial_rating
        self.k_const = k_const
        self.reset_rating_at_each_generation = reset_rating_at_each_generation
        self.ratings_df = pd.DataFrame()
        self.all_ratings = pd.DataFrame()
        self.recorded_points= pd.DataFrame()
        
        
    def add_ranking_model(
        self,
        model_id,
        initial_rating=None,
        generation_born=0,
        force_new_rating=False, 
    ):
        if model_id in self.models.keys():
            print("Model already present; not adding.")
            if self.reset_rating_at_each_generation:
                print("Resetting model's rating to {}".format(self.initial_rating))
                self.models[model_id].update_rating(
                    opponent_id=None, 
                    winner_id=None, 
                    new_rating=self.initial_rating,
                )
            elif (
                force_new_rating 
                and initial_rating > self.models[model_id].rating
            ):
                # move rating e.g. to align with others in batch
                print("Reassigning rating for model {} to {} points".format(model_id, initial_rating))
                self.models[model_id].update_rating(
                    opponent_id=None, 
                    winner_id=None, 
                    new_rating=initial_rating,
                )
            else: 
                print("NOT reassigning rating for model {}, as its current rating ({}) is higher than the proposed ({})".format(
                    model_id, 
                    self.models[model_id].rating,
                    initial_rating
                ))

            return
        if (
            self.reset_rating_at_each_generation
            or initial_rating is None 
        ):
            initial_rating = self.initial_rating

        print("Adding {} with starting rating {}".format(model_id, initial_rating))
        new_model = RateableModel(
            model_id = model_id, 
            initial_rating = initial_rating,
            generation_born = generation_born
        )
        self.models[model_id] = new_model
        
    def get_ratings(
        self,
        model_list=None
    ):
        for m in model_list: 
            if m not in self.models:
                self.add_ranking_model(model_id = m)

        self.ratings = {
            m : self.models[m].rating
            for m in self.models
        }
        if model_list is None:
            return self.ratings        
        else: 
            model_list_ratings = {
                m : self.ratings[m]
                for m in model_list
            }
            return model_list_ratings
        
    def get_rankings(self, model_list=None):
        ratings = self.get_ratings(model_list=model_list)
        rankings = sorted(
            ratings,
            key=ratings.get,
            reverse=True
        )
        return rankings

    def batch_update(
        self, 
        model_pairs_bayes_factors,
        spawn_step=0,
        force_new_rating=False,
    ):
        print("Ratings batch update for spawn step ", spawn_step)
        models = list(set(qmla.utilities.flatten(list(model_pairs_bayes_factors.keys()))))
        models_already_present = list(
            set(models).intersection(self.models)
        )
        print("rating df models:", self.models)
        print("Models:", models, "\n already present:",models_already_present)
        try:
            ratings = self.get_ratings(model_list=models_already_present)
            ratings = list(sorted( ratings.values(), reverse=True ))
            print("Ratings of present models", ratings)
            min_rating = ratings[1] # take the 2nd highest rating # TODO generalise
        except:
            min_rating = self.initial_rating

        for model in models:
            # add model to ratings database, 
            # starting with rating of worst model already present
            self.add_ranking_model(
                model_id = model, 
                initial_rating = min_rating,
                force_new_rating=force_new_rating,
            )

        for model_id in models:
            # update ratings df for plotting so 
            # start of next generation is end of previous one
            latest = pd.Series(
                {
                    'model_id' : model_id, 
                    'generation' : spawn_step,
                    'rating' : self.models[model_id].rating,
                    'idx' : 0                    
                }
            )
            self.all_ratings = self.all_ratings.append(
                latest, ignore_index=True
            )


        pairs = list(model_pairs_bayes_factors.keys())
        for a, b in pairs:
            self.compute_new_ratings(
                model_a_id = a,
                model_b_id = b, 
                bayes_factor = model_pairs_bayes_factors[a, b],
                spawn_step = spawn_step
            )


    def plot_models_ratings_against_generation(
        self, 
        f_scores,
        f_score_cmap,
        save_directory,
        show_fscore_cmap=False,
    ):

        all_model_ratings_by_generation = pd.DataFrame()
        models = self.all_ratings.model_id.unique()

        for model in models:
            model_ratings = self.all_ratings[ self.all_ratings.model_id == model ]
            generations = model_ratings.generation.unique()
            
            for g in generations:
                mod_ratings_this_generation = model_ratings[model_ratings.generation == g]
                
                start_idx = mod_ratings_this_generation.idx.min()
                final_idx = mod_ratings_this_generation.idx.max()
                
                start_rating = self.all_ratings[
                    (self.all_ratings.model_id == model)
                    & (self.all_ratings.generation == g)
                    & (self.all_ratings.idx == start_idx)
                ].rating.item()
                final_rating = self.all_ratings[
                    (self.all_ratings.model_id == model)
                    & (self.all_ratings.generation == g)
                    & (self.all_ratings.idx == final_idx)
                ].rating.item()        
                
                new_data = [
                    pd.Series({
                        'model_id' : model, 
                        'generation' : g, 
                        'rating' : start_rating
                    }),
                    pd.Series({
                        'model_id' : model, 
                        'generation' : g+0.8, 
                        'rating' : final_rating
                    }),
                ]
                
                for d in new_data:
                    all_model_ratings_by_generation = all_model_ratings_by_generation.append(
                        d, ignore_index=True
                    )

        # First prepare a dictionary to map model id to a colour corresponding to F-score
        f_granularity = 0.05
        available_f_scores = np.linspace(0, 1, 1 + (1/f_granularity) )

        model_coloured_by_f = {
            # m : colour_by_f[ qmla.utilities.round_nearest(f_scores[m], f_granularity) ]
            m : f_score_cmap(f_scores[m])
            for m in all_model_ratings_by_generation.model_id.unique()
        }

        # Plot
        widths = [1]
        if show_fscore_cmap:
            widths.append(0.1)
            legend_axis = (0,1)
        else:
            legend_axis = None
        lf = LatexFigure(
            # via https://github.com/flynnbr11/lfig-py
            use_gridspec=True, 
            gridspec_layout=(1,len(widths)),
            gridspec_params = {
                'width_ratios' : widths,
            },
            legend_axis=legend_axis
        )
        ax = lf.new_axis()
        sns.lineplot(
            x = 'generation', 
            y = 'rating', 
            hue = 'model_id', 
            data = all_model_ratings_by_generation, 
            palette=model_coloured_by_f,
            legend=False,
            ax = ax
        )
        for g in self.all_ratings.generation.unique():
            ax.axvline(g, ls='--', c='black')
        ax.axhline(self.initial_rating, ls=':', color='black')

        label_fontsize = 25
        ax.set_xlabel('Generation', 
            # fontsize = label_fontsize
        )
        ax.set_ylabel(
            r"$R$"
        )
        ax.set_xticks(list(self.all_ratings.generation.unique()))

        if show_fscore_cmap:
            ax = lf.legend_ax
            sm = plt.cm.ScalarMappable(
                cmap = f_score_cmap, 
                norm=plt.Normalize(vmin=0, vmax=1)
            )
            sm.set_array(available_f_scores)
            plt.colorbar(sm, cax=ax, orientation='vertical')
            ax.set_ylabel('F-score',  
                # fontsize=label_fontsize
            )

        lf.save(os.path.join(save_directory, 'elo_ratings_of_all_models.png'))

    def plot_rating_progress_single_model(
        self, 
        **kwargs
    ):
        self.plot_rating_progress_single_model_static(
            ratings_df = self.ratings_df, 
            **kwargs
        )

    @staticmethod
    def plot_rating_progress_single_model_static(
        ratings_df, 
        target_model_id, 
        return_df=False, 
        save_to_file=None
    ):

        # First isolate the ratings for this model
        model_identifiers = ["a", "b"]
        ratings_of_single_model = pd.DataFrame(
            columns=[
                'target', 'opponent', 
                'initial_rating_target', 'initial_rating_opponent', 
                'delta_r_target', 'final_rating_target',
                'idx', 
                'bayes_factor', 'generation', 'weight'
            ]
        )
        for target in model_identifiers:
            opponent = list(set(model_identifiers) - set(target))[0]

            target_ratings = ratings_df[
                ratings_df['model_{}'.format(target)] == target_model_id
                ][[
                    'model_{}'.format(target),
                    'model_{}'.format(opponent),
                    'r_{}_initial'.format(target),
                    'r_{}_initial'.format(opponent),
                    'delta_r_{}'.format(target),
                    'r_{}_new'.format(target),
                    'idx',
                    'bayes_factor',
                    'generation', 
                    'weight',
                    'winner'
                ]
            ]

            target_ratings.rename(
                columns={
                    'model_{}'.format(target) : 'target', 
                    'model_{}'.format(opponent) : 'opponent', 
                    'r_{}_initial'.format(target) : 'initial_rating_target',
                    'r_{}_initial'.format(opponent) : 'initial_rating_opponent', 
                    'r_{}_new'.format(target) : 'final_rating_target',
                    'delta_r_{}'.format(target) : 'delta_r_target'
                }, 
                inplace=True
            )

            ratings_of_single_model = ratings_of_single_model.append(target_ratings, ignore_index=True)

        ratings_of_single_model['won_comparison'] = (ratings_of_single_model.winner == ratings_of_single_model.target)
        ratings_of_single_model.sort_values('idx', inplace=True)
        ratings_of_single_model['new_idx'] = list(range(len(ratings_of_single_model)))
        
        # Plot 3 perspectives on this data:
        # 1. How rating changes
        # 2. relative change per comparison
        # 3. strength of evidence each comparison
        
        fig = plt.figure(
            figsize=(15, 10),
            constrained_layout=True,
        )
        gs = GridSpec(
            nrows=3,
            ncols=1,
            hspace=0.2,
        )

        # Actual ratings
        ax0 = plt.subplot(gs[0])
        sns.lineplot(
            data = ratings_of_single_model, 
            x = 'new_idx', 
            y = 'initial_rating_target',
            label='Initial',
            color='grey',
            ax = ax0,
        )

        sns.lineplot(
            data = ratings_of_single_model, 
            x = 'new_idx', 
            y = 'final_rating_target', 
            label='Final',
            color = 'blue', 
            ax = ax0
        )

        sns.scatterplot(
            data = ratings_of_single_model, 
            x = 'new_idx', 
            y = 'initial_rating_opponent',
            label='Opponent', 
            ax = ax0, 
        )
        ax0.legend()
        ax0.set_ylabel('Rating')
        ax0.set_xticks([])
        ax0.set_xlabel("")

        # Change in rating R
        ax1 = plt.subplot(gs[1], sharex = ax0)
        sns.barplot(
            data = ratings_of_single_model, 
            y = 'delta_r_target', 
            x = 'new_idx',
            hue='won_comparison',
            palette=['red', 'green'],
            ax = ax1
        )
        ax1.set_ylabel(r"$\Delta R$")
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.get_legend().remove()

        # Bayes factor comparisons
        ax2 = plt.subplot(gs[2], sharex = ax0)
        sns.barplot(
            data = ratings_of_single_model, 
            y = 'weight', 
            x = 'new_idx',
            hue='won_comparison',
            palette=['red', 'green'],
            ax = ax2
        )
        ax2.legend(title='Won')
        ax2.set_ylabel(r"$log_{10}(BF)$")

        generations = ratings_of_single_model.generation.unique()
        print("Generations:", generations)
        generation_change_indices = {
            g : ratings_of_single_model[
                ratings_of_single_model.generation == g].new_idx.max() + 0.5
            for g in generations
        }

        # vertical lines separating generations
        for ax in [ax0, ax1, ax2]:
            for g in generation_change_indices.values():
                ax.axvline(g,ls='--', c='grey', alpha=0.6, )

        # label x-axis with generations
        xtick_locations = [generation_change_indices[g] for g in generations]
        xtick_locations.insert(0,0)
        centred_xticks = [ 
            np.mean([xtick_locations[i], xtick_locations[i+1] ]) 
            for i in range(len(xtick_locations)-1) 
        ]
        ax2.set_xticklabels(generations)
        ax2.set_xticks(centred_xticks)
        ax2.set_xlabel("Generation")

        if save_to_file is not None: 
            fig.savefig(save_to_file)
        
        if return_df:
            return ratings_of_single_model


class RateableModel():
    def __init__(
        self, 
        model_id, 
        initial_rating=1000,
        generation_born = 0,
    ):
        self.model_id = model_id
        self.rating = initial_rating
        self.opponents_considered = []
        self.opponents_record = {}
        self.rating_history = {generation_born : [self.rating]}
        self.generation_born = generation_born

    def update_rating(
        self,
        opponent_id,
        winner_id, 
        new_rating, 
        generation=0, 
    ):
        # assumes the calculation has occured outside
        # this model class, and here we update the record
        self.opponents_considered.append(opponent_id)
        self.rating = new_rating
        if generation not in self.rating_history:
            self.rating_history[generation] = [self.rating]
        self.rating_history[generation].append(np.round(new_rating, 1))       

        if winner_id == self.model_id: 
            win = 1
        else: 
            win = 0
        try:
            self.opponents_record[opponent_id].append(win)
        except:
            self.opponents_record[opponent_id] = [win]
            
    @property
    def q_value(self):
        return 10**(self.rating/400)

class ELORating(RatingSystem):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
    def expected_score(
        self, 
        model_a_id, 
        model_b_id,
    ):
        rating_a = self.models[model_a_id].rating
        rating_b = self.models[model_b_id].rating
        p = (rating_a - rating_b)/400
        expected_score = 1.0 / (1 + 10**p)
        return expected_score
        
    def compute_new_ratings(
        self, 
        model_a_id, 
        model_b_id, 
        bayes_factor, 
        winner_id = None, 
        **kwargs
    ):
        if model_a_id not in self.models: 
            self.add_ranking_model(model_id = model_a_id)
        if model_b_id not in self.models: 
            self.add_ranking_model(model_id = model_b_id)
                        
        model_a = self.models[model_a_id]
        model_b = self.models[model_b_id]
        
        rating_a = model_a.rating
        rating_b = model_b.rating
                
        q_a = model_a.q_value
        q_b = model_b.q_value
        
        prob_a = q_a / (q_a + q_b)
        prob_b = q_b / (q_a + q_b)
        
        if winner_id == model_a_id: 
            rating_a_new = rating_a + (self.k_const * (1 - prob_a))
            rating_b_new = rating_b + (self.k_const * (0 - prob_b))
        elif winner_id == model_b_id: 
            rating_a_new = rating_a + (self.k_const * (0 - prob_a))
            rating_b_new = rating_b + (self.k_const * (1 - prob_b))
        rating_a_new = int(rating_a_new)
        rating_b_new = int(rating_b_new)

        model_a.update_rating(
            opponent_id = model_b_id, 
            winner_id = winner_id,
            new_rating = rating_a_new            
        )
        model_b.update_rating(
            opponent_id = model_a_id, 
            winner_id = winner_id,
            new_rating = rating_b_new            
        )
    


class ModifiedEloRating(ELORating):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_new_ratings(
        self, 
        model_a_id, 
        model_b_id, 
        bayes_factor, 
        spawn_step, 
        winner_id = None, 
        weight_log_base=10, 
        **kwargs
    ):
        if model_a_id not in self.models: 
            self.add_ranking_model(model_id = model_a_id, generation_born=spawn_step)
        if model_b_id not in self.models: 
            self.add_ranking_model(model_id = model_b_id, generation_born=spawn_step)

        model_a = self.models[model_a_id]
        model_b = self.models[model_b_id]

        if winner_id is None:
            if bayes_factor > 1 : 
                winner_id = model_a_id
            else:
                winner_id = model_b_id
        print("Rating update. A/B={}/{} \t BF={}".format(model_a_id, model_b_id, bayes_factor))

        rating_a = model_a.rating
        rating_b = model_b.rating
                
        q_a = model_a.q_value
        q_b = model_b.q_value
        prob_a = q_a / (q_a + q_b) # expectation A will win
        prob_b = q_b / (q_a + q_b) # expectation B will win
        
        if bayes_factor > 1:
            # bayes_factor_weight = np.log10(bayes_factor)
            bayes_factor_weight = math.log(bayes_factor, weight_log_base)

        else:
            # bayes_factor_weight = np.log10(1/bayes_factor)
            bayes_factor_weight = math.log(1/bayes_factor, weight_log_base)

        # update A
        if winner_id == model_a_id: 
            result_a = 1 # A won
            result_b = 0
        else:
            result_a = 0 # A lost
            result_b = 1
        delta_a = bayes_factor_weight * (result_a - prob_a)
        delta_b = bayes_factor_weight * (result_b - prob_b)

        rating_a_new = np.round(rating_a + delta_a, 2)
        rating_b_new = np.round(rating_b + delta_b, 2)

        model_a.update_rating(
            opponent_id = model_b_id, 
            winner_id = winner_id,
            new_rating = rating_a_new,
            generation = spawn_step,      
        )
        model_b.update_rating(
            opponent_id = model_a_id, 
            winner_id = winner_id,
            new_rating = rating_b_new,
            generation = spawn_step,      
        )


        this_round = pd.Series({
            'model_a' : model_a_id, 
            'model_b' : model_b_id, 
            # r'$R^{a}_{0}$'  : rating_a, 
            # r'$R^{b}_{0}$' : rating_b,
            # r'$R^{a}_{new}$' : rating_a_new, 
            # r'$R^{b}_{new}$' : rating_b_new,
            # r"$\Delta R^{a}$" : delta_a,
            # r"$\Delta R^{b}$" : delta_b,
            'r_a_initial'  : rating_a, 
            'r_b_initial' : rating_b,
            'r_a_new' : rating_a_new, 
            'r_b_new' : rating_b_new,
            'delta_r_a' : delta_a,
            'delta_r_b' : delta_b,
            'bayes_factor' : bayes_factor,
            'weight' : bayes_factor_weight,
            'winner' : winner_id,
            'generation' : spawn_step, 
            'idx' : len(self.ratings_df) # to track the order in which these ratings are recorded
        })
        
        for mod in [model_a, model_b]:
            new_idx = len(
                self.all_ratings[ (self.all_ratings.model_id == mod.model_id) 
                & (self.all_ratings.generation == spawn_step)]
            )
            latest = pd.Series(
                {
                    'model_id' : mod.model_id, 
                    'generation' : spawn_step,
                    'rating' : mod.rating,
                    'idx' : new_idx,
                    
                }
            )
            self.all_ratings = self.all_ratings.append(
                latest, ignore_index=True
            )
        
        self.ratings_df = self.ratings_df.append(
            this_round, 
            ignore_index=True
        )

