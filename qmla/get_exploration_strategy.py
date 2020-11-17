from __future__ import absolute_import
import inspect

import qmla.exploration_strategies as GR

__all__ = [
    'exploration_classes',
    'get_exploration_class'
]

# Get a dict of all the available exploration strategies
exploration_classes = dict(inspect.getmembers(GR, inspect.isclass))

def get_exploration_class(
    exploration_rules,
    **kwargs
):
    r"""
    Get an instance of the class specified by the user which implements an exploration Strategy.

    Instance of a :class:`~qmla.ExplorationStrategy` (or subclass).
    This is used to specify how QMLA proceeds, in particular by designing the next batch
    of models to test.
    Exploration Strategy is specified by the name passed to implement_qmla in the launch script,
    through the command line flag `exploration_strategy`. This string is searched for in the
    exploration_classes dictionary. New exploration strategies must be added here so that QMLA can find them.


    :param str exploration_rules: string corresponding to an exploration strategy
    :params **kwargs: arguments required by the exploration strategy, passed directly
        to the desired exploration strategy's constructor.
    :return ExplorationStrategy gr: exploration strategy class instance
    """

    try:
        gr = exploration_classes[exploration_rules](
            exploration_rules=exploration_rules,
            **kwargs
        )
    except BaseException:
        print(
            "Cannot find exploration strategy in available rules:",
            exploration_rules)
        raise

    return gr
