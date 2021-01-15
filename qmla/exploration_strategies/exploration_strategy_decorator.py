r"""
    This is an initial effort towards using decorators with exploration strategies, 
    which is incomplete and not currently in use anywhere. 
"""


class ExplorationStrategyDecorator():
    def __init__(self, exploration_strategy, **kwargs):
        print("exploration strategy decorator __init__. exploration strategy: ", exploration_strategy)
        print("kwargs: ", kwargs)
        self.exploration_strategy = exploration_strategy
        print("assigned self.exploration_strategy")


    def __call__(
        self, 
        exploration_rules, 
        # configuration = None,  
        # *args, 
        **kwargs
    ):
        print("exploration strategy decorator __call__. kwargs: ", kwargs )
        
        gr = self.exploration_strategy(
            exploration_rules = exploration_rules,
            # *args, 
            **kwargs
        )
        
        # # some protected attributes 
        # # unique to this instance, so reassign manually 
        # log_file = gr.log_file

        # if 'configuration' in kwargs and kwargs['configuration'] is not None:
        #     gr.__dict__ = kwargs['configuration']
        #     gr.__dict__['log_file'] = log_file
        # else: 
        #     gr.assign_parameters()
            
        return gr
        


# class ExplorationStrategyDecorator():
#     def __init__(self, exploration_strategy):
#         print("exploration strategy wrapper __init__")
#         self.exploration_strategy = exploration_strategy

#     def __call__(self, *args, **kwargs):
#         print("exploration strategy wrapper __call__")
        
#         gr = self.exploration_strategy(*args, **kwargs)
        
#         # some protected attributes 
#         # unique to this instance, so reassign manually 
#         log_file = gr.log_file

#         if 'configuration' in kwargs and kwargs['configuration'] is not None:
#             gr.__dict__ = kwargs['configuration']
#             gr.__dict__['log_file'] = log_file
#         else: 
#             gr.assign_parameters()
            
#         return gr
        
