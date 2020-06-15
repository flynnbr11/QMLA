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

import qmla.utilities

class RatingSystem():
    def __init__(
        self,
        initial_rating=1000,
        k_const=30 # constant in calculation of ELO rating (?)
    ):
        self.models = {}
        self.initial_rating = initial_rating
        self.k_const = k_const
        self.ratings_df = pd.DataFrame()
        self.all_ratings = pd.DataFrame()
        self.recorded_points= pd.DataFrame()
        
        
    def add_ranking_model(
        self,
        model_id,
        initial_rating=None,
        generation_born=0,
    ):
        if model_id in self.models.keys():
            print("Model already present; not adding.")
            return
        print("Adding {} with starting rating {}".format(model_id, initial_rating))
        if initial_rating is None:
            initial_rating = self.initial_rating
        new_model = RateableModel(
            model_id = model_id, 
            initial_rating = initial_rating,
            generation_born = generation_born
        )
        self.models[model_id] = new_model
        # latest = pd.Series({
        #     'model_id' : new_model.model_id, 
        #     'generation' : generation_born,
        #     'rating' : new_model.rating,
        #     'idx' : 0
        # })
        # self.all_ratings = self.all_ratings.append(
        #     latest, ignore_index=True
        # )

        
    def get_ratings(
        self,
        model_list=None
    ):
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
            ratings = list(sorted( ratings.values() ))
            print("Ratings of present models", ratings)
            min_rating = ratings[1] # take the 2nd highest rating # TODO generalise
        except:
            min_rating = self.initial_rating

        for model in models:
            # add model to ratings database, 
            # starting with rating of worst model already present
            self.add_ranking_model(
                model_id = model, 
                initial_rating = min_rating
            )

        for model_id in models:
            # update ratings df for plotting so 
            # start of next generation is end of previous one
            # new_idx = len(
            #     self.all_ratings[ (self.all_ratings.model_id == model_id) 
            #     & (self.all_ratings.generation == spawn_step)]
            # )
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
        save_directory,
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
        f_score_colour_map = plt.cm.Spectral

        available_f_scores = np.linspace(0, 1, 1 + (1/f_granularity) )
        my_cmap = f_score_colour_map(available_f_scores)

        # f_score_cmap = plt.cm.get_cmap('Blues')
        # f_score_cmap = qmla.utilities.truncate_colormap(f_score_cmap, 0.25, 1.0)
        # f_score_cmap = plt.cm.get_cmap('tab20c_r')
        # f_score_cmap = plt.cm.get_cmap('Accent')
        f_score_cmap = matplotlib.colors.ListedColormap(["sienna", "red", "darkorange", "gold", "blue"])

        # f_score_cmap = qmla.utilities.truncate_colormap(f_score_cmap, 0.6, 1.0)



        # colour_by_f = {
        #     # np.round(f, 2) : my_cmap[ np.where( available_f_scores == f ) ][0]
        #     np.round(f, 2) : f_score_cmap[ np.where( available_f_scores == f ) ][0]
        #     for f in available_f_scores
        # }

        model_coloured_by_f = {
            # m : colour_by_f[ qmla.utilities.round_nearest(f_scores[m], f_granularity) ]
            m : f_score_cmap(f_scores[m])
            for m in all_model_ratings_by_generation.model_id.unique()
        }

        # Plot
        fig, ax = plt.subplots(figsize=(15,10), constrained_layout=True)
        gs = GridSpec(
            nrows=1, ncols=2,
            width_ratios=[10,1]
        )

        ax = fig.add_subplot(gs[0,0])
        sns.lineplot(
            x = 'generation', 
            y = 'rating', 
            hue = 'model_id', 
            data = all_model_ratings_by_generation, 
            palette=model_coloured_by_f,
            legend=False
        )
        for g in self.all_ratings.generation.unique():
            ax.axvline(g, ls='--', c='black')
        ax.axhline(self.initial_rating, ls=':', color='black')

        label_fontsize = 25
        ax.set_xlabel('Generation', fontsize = label_fontsize)
        ax.set_ylabel('Modified Elo rating', fontsize=label_fontsize)
        ax.set_xticks(list(self.all_ratings.generation.unique()))

        # color bar
        ax = fig.add_subplot(gs[0,1])
        sm = plt.cm.ScalarMappable(
            # cmap = f_score_colour_map, 
            cmap = f_score_cmap, 
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array(available_f_scores)

        
        plt.colorbar(sm, cax=ax, orientation='vertical')
        ax.set_ylabel('F-score',  fontsize=label_fontsize)

        fig.savefig(
            os.path.join(save_directory, 'elo_ratings_of_all_models.png')
        )


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
            new_rating = rating_a_new       ,
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
            r'$R^{a}_{0}$'  : rating_a, 
            r'$R^{b}_{0}$' : rating_b,
            r'$R^{a}_{new}$' : rating_a_new, 
            r'$R^{b}_{new}$' : rating_b_new,
            r"$\Delta R^{a}$" : delta_a,
            r"$\Delta R^{b}$" : delta_b,
            'bayes_factor' : bayes_factor,
            'weight' : bayes_factor_weight,
            'winner' : winner_id,
            'generation' : spawn_step, 
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
                    'idx' : new_idx
                    
                }
            )
            self.all_ratings = self.all_ratings.append(
                latest, ignore_index=True
            )
        
        self.ratings_df = self.ratings_df.append(
            this_round, 
            ignore_index=True
        )
