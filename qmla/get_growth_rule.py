from __future__ import absolute_import
import inspect

import qmla.growth_rules as GR

__all__ = [
    'growth_classes',
    'get_growth_generator_class'
]

# Get a dict of all the available growth rules
growth_classes = dict(inspect.getmembers(GR, inspect.isclass))

def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    r"""
    Get an instance of the class specified by the user which implements a Growth Rule.

    Instance of a :class:`~qmla.GrowthRule` (or subclass).
    This is used to specify how QMLA proceeds, in particular by designing the next batch
    of models to test.
    Growth rule is specified by the name passed to implement_qmla in the launch script,
    through the command line flag `growth_rule`. This string is searched for in the
    growth_classes dictionary. New growth rules must be added here so that QMLA can find them.


    :param str growth_generation_rule: string corresponding to a growth rule
    :params **kwargs: arguments required by the growth rule, passed directly
        to the desired growth rule's constructor.
    :return GrowthRule gr: growth rule class instance
    """

    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
    except BaseException:
        print(
            "Cannot find growth rule in available rules:",
            growth_generation_rule)
        raise

    return gr
