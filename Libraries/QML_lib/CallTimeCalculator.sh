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

python3 CalculateTimeRequired.py \
	-ggr='heisenberg_xyz' \
	$additional_growth \
	-use_agr=1 \
	-e=400 \
	-p=1500 \
	-bt=425 \
	-proc=1 \
	-res=0 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME"
