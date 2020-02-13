
class GrowthRuleDecorator():
    def __init__(self, growth_rule):
        print("growth rule wrapper __init__")
        self.growth_rule = growth_rule

    def __call__(
        self, 
        growth_generation_rule, 
        # configuration = None,  
        # *args, 
        **kwargs
    ):
        print("growth rule wrapper __call__")
        
        gr = self.growth_rule(
            growth_generation_rule = growth_generation_rule,
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
        


# class GrowthRuleDecorator():
#     def __init__(self, growth_rule):
#         print("growth rule wrapper __init__")
#         self.growth_rule = growth_rule

#     def __call__(self, *args, **kwargs):
#         print("growth rule wrapper __call__")
        
#         gr = self.growth_rule(*args, **kwargs)
        
#         # some protected attributes 
#         # unique to this instance, so reassign manually 
#         log_file = gr.log_file

#         if 'configuration' in kwargs and kwargs['configuration'] is not None:
#             gr.__dict__ = kwargs['configuration']
#             gr.__dict__['log_file'] = log_file
#         else: 
#             gr.assign_parameters()
            
#         return gr
        
