#!/bin/bash

test_description="run_20_prt_3000_one_core_four_ppn"

echo "local host is $(hostname). Global redis launced here." 
# ./global_redis_launch.sh

this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)
#results_dir=$dir_name/Results/$day_time

script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
results_dir=$day_time
full_path_to_results=$script_dir/Results/$results_dir


mkdir -p results_dir

global_server=$(hostname)

for i in `seq 3 3`;
do
	this_qmd_name="$test_description""_$i"
	echo "This name: $this_qmd_name"
	qsub -v QMD_ID=$i,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir -N $this_qmd_name launch_qmd_parallel.sh

done 

echo "
#!/bin/bash 
cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results"
" > analyse_$test_description.sh
