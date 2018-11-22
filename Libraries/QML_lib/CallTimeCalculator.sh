#!/bin/bash

RESULTS_DIR='../../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"
# rm $script_path
python3 CalculateTimeRequired.py \
	-ggr='deterministic_transverse_ising_nn_fixed_axis' \
	-e=1500 \
	-p=3000 \
	-bt=1500 \
	-proc=5 \
	-res=0 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME"
