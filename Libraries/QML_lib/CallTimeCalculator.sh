#!/bin/bash

RESULTS_DIR='../../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"
# rm $script_path
python3 CalculateTimeRequired.py \
	-ggr='deterministic_interacting_nn_ising_single_axis' \
	-e=1500 \
	-p=3000 \
	-bt=1500 \
	-proc=5 \
	-res=0 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME"
	# -ggr=$GGR \
	# -e=$EXPERIMENTS \
	# -p=$PARTICLES \
	# -bt=BAYES_TIMES \
	# -proc=$NUM_PROC \
	# -res=$RESCALE_RESOURCES \
	# -scr=$script_path \
	# -qmdtenv=$QMD_TIME \
	# -qhltenv=$QHL_TIME


# chmod a+x $script_path
# source $script_path
