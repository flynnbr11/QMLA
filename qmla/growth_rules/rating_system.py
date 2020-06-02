import sys
import os
import numpy as np
import random
import pandas as pd


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
        
        
    def add_ranking_model(
        self,
        model_id
    ):
        new_model = RateableModel(
            model_id = model_id, 
            initial_rating = self.initial_rating
        )
        self.models[model_id] = new_model

        
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
        

class RateableModel():
    def __init__(
        self, 
        model_id, 
        initial_rating=1000
    ):
        self.model_id = model_id
        self.rating = initial_rating
        self.opponents_considered = []
        self.opponents_record = {}
        self.rating_history = [self.rating]

    def update_rating(
        self,
        opponent_id,
        winner_id, 
        new_rating, 
    ):
        # assumes the calculation has occured outside
        # this model class, and here we update the record
        self.opponents_considered.append(opponent_id)
        self.rating = new_rating
        self.rating_history.append(np.round(new_rating, 1))
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
        winner_id, 
        bayes_factor, 
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
        winner_id, 
        bayes_factor, 
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
        prob_a = q_a / (q_a + q_b) # expectation A will win
        prob_b = q_b / (q_a + q_b) # expectation B will win
        
        if bayes_factor > 1:
            bayes_factor_weight = np.log10(bayes_factor)
        else:
            bayes_factor_weight = np.log10(1/bayes_factor)

        if winner_id == model_a_id: 
            rating_a_new = rating_a + (bayes_factor_weight * (1 - prob_a))
            rating_b_new = rating_b + (bayes_factor_weight * (0 - prob_b))
        elif winner_id == model_b_id: 
            rating_a_new = rating_a + (bayes_factor_weight * (0 - prob_a))
            rating_b_new = rating_b + (bayes_factor_weight * (1 - prob_b))
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

        this_round = pd.Series({
            'model_a' : model_a_id, 
            'model_b' : model_b_id, 
            'a_rating_before'  : rating_a, 
            'b_rating_before' : rating_b,
            'a_rating_after' : rating_a_new, 
            'b_rating_new' : rating_b_new,
            'bayes_factor' : bayes_factor,
            'weight' : bayes_factor_weight
        })
        self.ratings_df = self.ratings_df.append(
            this_round, 
            ignore_index=True
        )
