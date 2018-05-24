#!/bin/bash

test_description="timing_tests"


num_tests=1
min_id=2
let max_id="$min_id + $num_tests - 1 "

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

very_short_time="walltime=00:06:00"
short_time="walltime=00:20:00"
medium_time="walltime=01:00:00"
long_time="walltime=08:00:00"
very_long_time="walltime=16:00:00"

test_time="walltime=00:90:00"

time=$test_time
qmd_id=0


NUM_PARTICLES=10
NUM_EXP=5
NUM_BAYES=4
RESAMPLE_A=0.95
RESAMPLE_T=0.5
RESAMPLE_PGH=1.0


for ((e=50; e<=100; e+=50));
do 
	for ((p=100; p<=300; p+=200));
	do 
		for ra in `seq 0.8 0.05 1.0 `;
		do
			for rt in `seq 0.35 0.05 0.65 `;
			do
				for rp in `seq 0.8 0.1 1.2`;
				do		
					for i in `seq $min_id $max_id`;
					do
						let bt="$e/2"
						let qmd_id="$qmd_id+1"
						let ham_exp="$e*$p + $p*$bt"
						let seconds_reqd="$ham_exp/50"
						time="walltime=00:00:$seconds_reqd"
						this_qmd_name="$test_description""_$qmd_id"
						echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
						qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
					done 
				done
			done
		done
	done

done




echo "
#!/bin/bash 
cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results"
" > analyse_$test_description.sh
