#!/bin/bash

RESULTS_DIR='../../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"
# rm $script_path

additional_growth="-agr=non_interacting_ising"
# additional_growth=""

#	-ggr='two_qubit_ising_rotation_hyperfine_transverse' \
python3 CalculateTimeRequired.py \
	-ggr='two_qubit_ising_rotation_hyperfine_transverse' \
	$additional_growth \
	-use_agr=0 \
	-e=500 \
	-p=2000 \
	-bt=1500 \
	-proc=5 \
	-res=0 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME"
