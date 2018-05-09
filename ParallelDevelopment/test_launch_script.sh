#!/bin/bash

echo "local host is $(hostname). Global redis launced here." 
# ./global_redis_launch.sh

this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)
#results_dir=$dir_name/Results/$day_time
results_dir=$day_time

mkdir -p results_dir

global_server=$(hostname)

for i in `seq 1 30`;
do
	qsub -v QMD_ID=$i,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir  launch_qmd_parallel.sh
done 


