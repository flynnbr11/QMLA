#!/bin/bash

RESULTS_DIR='../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"

# additional_es="-agr=IsingGenetic -agr=hubbard_square_lattice_generalised"
additional_es=""

exploration_strategy='VariableTrueModelNVCentre'
num_experiments=1000
num_particles=3000


python3 time_required_calculation.py \
	-es=$exploration_strategy \
	$additional_es \
	-use_aes=0 \
	-e=$num_experiments \
	-p=$num_particles \
	-proc=1 \
	-res=0 \
	-time_insurance=1 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME" \
	-num_proc_env="NUM_PROC"

echo "TIME requested $QMD_TIME" 