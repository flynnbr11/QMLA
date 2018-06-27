#!/bin/bash

test_description="wider_initial_dist"



num_tests=3
min_id=1
let max_id="$min_id + $num_tests - 1 "


this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)

script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
results_dir=$day_time
# full_path_to_results=$script_dir/Results/$results_dir
full_path_to_results=$(pwd)/Results/$results_dir
all_qmd_bayes_csv="$full_path_to_results/multiQMD.csv"


OUT_LOG="$(pwd)/logs/$day_time/OUTPUT_AND_ERROR_FILES/"
echo "pwd: $(pwd)"
echo "OUT LOG: $OUT_LOG"
output_file="output_file"
error_file="error_file" 

mkdir -p $full_path_to_results
mkdir -p "$(pwd)/logs"
mkdir -p $OUT_LOG
mkdir -p results_dir

global_server=$(hostname)

very_short_time="walltime=00:06:00"
short_time="walltime=00:20:00"
medium_time="walltime=01:00:00"
long_time="walltime=08:00:00"
very_long_time="walltime=16:00:00"

test_time="walltime=00:90:00"

time=$test_time
qmd_id=$min_id
cutoff_time=180

do_plots=1
pickle_class=0

p_min=3000
p_max=3000
p_int=1000
p_default=2000

e_min=2000 
e_max=2000
e_int=1000
e_default=2000

ra_min=0.8
ra_max=0.95
ra_int=0.05
ra_default=0.9

rt_min=0.4
rt_max=0.6
rt_int=0.1
rt_default=0.5

rp_min=0.9
rp_max=1.1
rp_int=0.1
rp_default=1.0


e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

e=5
p=10

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
	this_qmd_name="$test_description""_$qmd_id"
	this_error_file="$OUT_LOG/$error_file""_$qmd_id.txt"
	this_output_file="$OUT_LOG/$output_file""_$qmd_id.txt"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"

	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv -N $this_qmd_name -l $time -o $this_output_file -e $this_error_file run_qmd_instance.sh

done


echo "
#!/bin/bash 
cd $lib_dir
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv
" > $full_path_to_results/ANALYSE_$test_description.sh
