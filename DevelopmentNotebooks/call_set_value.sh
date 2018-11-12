#!/bin/bash

script='new_script.sh'

python3 set_value.py \
	-ggr='two_qubit_ising_rotation_hyperfine' \
	-e=10 \
	-p=20 \
	-bt=10 \
	-proc=2 \
	-res=0 \
	-scr=$script \
	-qtenv="QMD_TIME"


chmod a+x $script
source $script
