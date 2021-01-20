from __future__ import absolute_import
import sys
import os
import random

from qmla.exploration_strategies.nv_centre_spin_characterisation.nature_physics_2021 import FullAccessNVCentre
from qmla.construct_models import alph

__all__ = [
    'TieredGreedySearchNVCentre'
]

class TieredGreedySearchNVCentre(
    FullAccessNVCentre
):
    r"""
    Exploration strategy for NV system described in Nature Physics 2021 paper, 
    assuming full access to the state so the likelihood is based on 
    :math:`\langle++| e^{ -i\hat{H(\vec{x})} t } |++\rangle`. 

    This is the base class for results presented in the experimental paper, 
    namely Fig 2. 
    The same model generation strategy is used in each case (i), (ii), (iii):
        this ES is for (i) pure simulation. 

    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.initial_models = None
        self.term_tiers = {
            1 : ['xTi', 'yTi', 'zTi'],
            2 : ['xTx', 'yTy', 'zTz'],
            3 : ['xTy', 'xTz', 'yTz']
        }
        self.tier = 1
        self.max_tier = max(self.term_tiers)
        self.tier_branch_champs = {k : [] for k in self.term_tiers} 
        self.tier_champs = {}
        self.prune_completed_initially = True
        self.check_champion_reducibility = True

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        r""" 
        Overwrites :meth:`qmla.QuantumModelLearningAgent.generate_models`. 

        Constructs models in tiers, where each tier is explored greedily, and only the 
        strongest model from the tier is progressed as the seed model for the subsequent tier.
        """

        self.log_print([
            "Generating models in tiered greedy search at spawn {}. available kwargs:\n {}".format(
                self.spawn_step, kwargs
            )
        ])
        # self.spawn_step = kwargs['spawn_step']
        if self.spawn_stage[-1] is None:
            try:
                previous_branch_champ = model_list[0]
                self.tier_branch_champs[self.tier].append(previous_branch_champ)
            except:
                previous_branch_champ = None

        elif "getting_tier_champ" in self.spawn_stage[-1]:
            previous_branch_champ = model_list[0]
            self.log_print([
                "Tier champ for {} is {}".format(self.tier, model_list[0])
            ])
            self.tier_champs[self.tier] = model_list[0]
            self.tier += 1
            self.log_print(["Tier now = ", self.tier])
            self.spawn_stage.append(None) # normal processing

            if self.tier > self.max_tier:
                self.log_print(["Completed tree for ES"])
                self.spawn_stage.append('Complete')
                return list(self.tier_champs.values())
        else:
            self.log_print([
                "Spawn stage:", self.spawn_stage
            ])

        new_models = greedy_add(
            current_model = previous_branch_champ, 
            terms = self.term_tiers[self.tier]
        )
        self.log_print([
            "tiered search new_models=", new_models
        ])

        if len(new_models) == 0:
            # no models left to find - get champions of branches from this tier
            new_models = self.tier_branch_champs[self.tier]
            self.log_print([
                "tier champions: {}".format(new_models)
            ])
            self.spawn_stage.append("getting_tier_champ_{}".format(self.tier))
        return new_models

    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        r"""
        QMLA asks the exploration tree whether it has finished growing; 
        the exploration tree queries the exploration strategy through this method
        """
        if self.tree_completed_initially:
            return True
        elif self.spawn_stage[-1] == "Complete":
            return True
        else:
            return False

    def check_tree_pruned(self, prune_step, **kwargs):
        return self.prune_completed_initially

def greedy_add(
    current_model, 
    terms,
):
    r"""
    Generate a list of models by appending all individual terms to the current model. 
    """
    
    try:
        present_terms = current_model.split('+')
    except:
        present_terms = []
    nonpresent_terms = list(set(terms) - set(present_terms))
    
    term_sets = [
        present_terms+[t] for t in nonpresent_terms
    ]

    new_models = ["+".join(term_set) for term_set in term_sets]
    
    return new_models