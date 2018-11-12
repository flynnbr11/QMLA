#!/bin/bash

RESULTS_DIR='../../DevelopmentNotebooks'
script_name="call_time_script.sh"
script_path="$RESULTS_DIR/$script_name"
# rm $script_path
echo "results dir: $RESULTS_DIR"
echo "test input: $TEST_VAR"
echo "script path: $script_path"
python3 CalculateTimeRequired.py \
	-ggr='two_qubit_ising_rotation_hyperfine' \
	-e=10 \
	-p=20 \
	-bt=10 \
	-proc=2 \
	-res=0 \
	-scr=$script_path \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME"
	# -ggr=$GGR \
	# -e=$EXPERIMENTS \
	# -p=$PARTICLES \
	# -bt=BAYES_TIMES \
	# -proc=$NUM_PROC \
	# -res=$RESCALE_RESOURCES \
	# -scr=$script_path \
	# -qmdtenv=$QMD_TIME \
	# -qhltenv=$QHL_TIME


chmod a+x $script_path
source $script_path
