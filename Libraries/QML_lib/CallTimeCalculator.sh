#!/bin/bash

RESULTS_DIR='../../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"
# rm $script_path

additional_growth="-agr=ising_1d_chain -agr=hubbard_square_lattice_generalised"
# additional_growth=""

# -ggr='two_qubit_ising_rotation_hyperfine' \
# -ggr='two_qubit_ising_rotation_hyperfine_transverse' \

# -ggr='ising_1d_chain' \
# -ggr='hubbard_square_lattice_generalised' \

growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
num_experiments=10
num_particles=10


python3 CalculateTimeRequired.py \
	-ggr=$growth_rule \
	$additional_growth \
	-use_agr=0 \
	-e=$num_experiments \
	-p=$num_particles \
	-bt=425 \
	-proc=1 \
	-res=0 \
	-time_insurance=1 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME"
