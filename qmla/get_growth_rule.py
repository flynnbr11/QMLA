from __future__ import absolute_import

import sys
import os

import qmla.growth_rules as GR

growth_classes = {
    # # Experimental paper growth rules
    'two_qubit_ising_rotation_hyperfine_transverse':
        GR.nv_centre_spin_experimental_method,
    'two_qubit_ising_rotation_hyperfine':
        GR.nv_centre_spin_experimental_method_without_transvere_terms,
    'NV_alternative_model':
        GR.nv_centre_spin_experimental_method_alternative_true_model,
    'NV_alternative_model_2':
        GR.nv_centre_spin_experimental_method_alternative_true_model_second,
    'nv_experiment_vary_model':
        GR.nv_centre_spin_experimental_method_varying_true_model,
    'NV_fitness_growth':
        GR.nv_fitness_growth,
    'NV_centre_revivals':
        GR.nv_centre_revival_data,
    'NV_spin_full_access':
        GR.nv_centre_spin_full_access,
    'NV_centre_spin_large_bath':
        GR.nv_centre_large_spin_bath,
    'reduced_nv_experiment':
        GR.reduced_nv_experiment,
    # Theoretical paper growth rules
    'ising_predetermined':
        GR.IsingPredetermined,
    'ising_probabilistic':
        GR.IsingProbabilistic,
    'heisenberg_xyz_predetermined':
        GR.heisenberg_xyz_predetermined,
    'heisenberg_xyz_probabilistic':
        GR.heisenberg_xyz_probabilistic,
    'fermi_hubbard_predetermined':
        GR.FermiHubbardPredetermined,
    'fermi_hubbard_probabilistic':
        GR.FermiHubbardProbabilistic,
    # Others
    'basic_lindbladian':
        GR.basic_lindbladian,
    'example':
        GR.GrowthRuleTemplate,
    'presentation':
        GR.presentation_plot_generation,
    'genetic':
        GR.genetic_algorithm
}


def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    # TODO: note that in most places, this is called with use_experimental_data.
    # in some plotting functions this is not known, but it should not matter unless
    # called to get probes etc.

    # print("Trying to find growth class for ", growth_generation_rule)
    # print("kwargs:", kwargs)
    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
    except BaseException:
        raise
        # print("{} growth class not found.".format(growth_generation_rule))
        gr = None

    return gr
