#!/bin/bash

RESULTS_DIR='../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"

additional_growth="-agr=ising_1d_chain -agr=hubbard_square_lattice_generalised"
# additional_growth=""

# -ggr='two_qubit_ising_rotation_hyperfine' \
# -ggr='two_qubit_ising_rotation_hyperfine_transverse' \

# -ggr='ising_1d_chain' \
# -ggr='hubbard_square_lattice_generalised' \

# growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
# growth_rule='IsingLatticeSet'
growth_rule='HeisenbergLatticeSet'
num_experiments=250
num_particles=750


python3 time_required_calculation.py \
	-ggr=$growth_rule \
	$additional_growth \
	-use_agr=0 \
	-e=$num_experiments \
	-p=$num_particles \
	-bt=$num_experiments \
	-proc=1 \
	-res=0 \
	-time_insurance=1 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME" \
	-num_proc_env="NUM_PROC"


echo "TIME requested $QMD_TIME" 