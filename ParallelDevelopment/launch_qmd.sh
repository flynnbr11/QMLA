#!/bin/bash

test_description="qhl_test"

## QMD parameters
num_tests=2
do_plots=1
pickle_class=0
qhl=0 # do a test on QHL only -> 1; QMD -> 0
experimental_data=1 # use experimental data -> 1; use fake data ->0
custom_prior=1
#dataset='NV05_HahnPeaks_expdataset'
dataset='NVB_HahnPeaks_Newdata'
data_max_time=5000 # nanoseconds
data_time_offset=205 # nanoseconds
top_number_models=2 # how many models to perform further QHL for
do_further_qhl=1
further_qhl_resource_factor=2


p=20 # particles
e=8 # experiments
ra=0.8 #resample a 
rt=0.6 # resample threshold
rp=1.0 # PGH factor

true_hamiltonian='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'

### --------------------------------------------------- ###


### ---------------------------------------------------###
min_id=10 # update so instances don't clash and hit eachother's redis databases

let max_id="$min_id + $num_tests - 1 "


this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)

script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
results_dir=$day_time
# full_path_to_results=$script_dir/Results/$results_dir
full_path_to_results=$(pwd)/Results/$results_dir
all_qmd_bayes_csv="$full_path_to_results/multiQMD.csv"


OUT_LOG="$full_path_to_results/OUTPUT_AND_ERROR_FILES/"
output_file="output_file"
error_file="error_file" 

mkdir -p $full_path_to_results
mkdir -p "$(pwd)/logs"
mkdir -p $OUT_LOG
mkdir -p results_dir

global_server=$(hostname)
test_time="walltime=00:90:00"

time=$test_time
qmd_id=$min_id
cutoff_time=360

echo "" > $full_path_to_results/job_ids_started.txt
echo "" > $full_path_to_results/job_ids_completed.txt


declare -a qhl_operators=(
'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
)

declare -a resample_a_values=(
$ra
)

declare -a resample_thresh_values=(
$rt
)

declare -a pgh_values=(
$rp
)

declare -a particle_counts=(
$p
)

declare -a experiment_counts=(
$e
)


printf "$day_time: \t $test_description \t e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp \n" >> QMD_Results_directories.log
num_jobs_launched=0

for rp in "${pgh_values[@]}";
do
	for rt in "${resample_thresh_values[@]}";
	do
		for ra in "${resample_a_values[@]}";
		do

			for p in  "${particle_counts[@]}";
			do
				for e in "${experiment_counts[@]}";
				do
					for i in `seq $min_id $max_id`;
					do
						let bt="$e-1"
						let qmd_id="$qmd_id+1"
						let ham_exp="$e*$p + $p*$bt"
						let expected_time="$ham_exp/30"
						let num_jobs_launched="$num_jobs_launched+1"
						if [ "$qhl" == 1 ]
						then
							let expected_time="$expected_time/10"
						fi
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
						printf "$day_time: \t e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd \n" >> QMD_all_tasks.log

						qsub -v QMD_ID=$qmd_id,OP="$true_hamiltonian",QHL=$qhl,FURTHER_QHL=0,EXP_DATA=$experimental_data,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset -N $this_qmd_name -l $time -o $this_output_file -e $this_error_file run_qmd_instance.sh

					done
				done
			done
		done
	done
done

finalise_qmd_script=$full_path_to_results/FINALISE_$test_description.sh
monitor_script=$full_path_to_results/monitor.sh

### Analyse results of QMD runs. 
echo "
#!/bin/bash 
cd $lib_dir
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv -top=$top_number_models
" > $finalise_qmd_script

### Further QHL on top performing models.
if (( "$do_further_qhl" == 1 ))
then
	let p="$further_qhl_resource_factor*$p"
	let e="$further_qhl_resource_factor*$e"
	let bt="$e-1"
	rp=1.0
	let qmd_id="2+$qmd_id"
	pbs_config=10:00:00,nodes=1:ppn=$top_number_models

	echo "
cd $(pwd)

qsub -v QMD_ID=$qmd_id,OP="$true_hamiltonian",QHL=$qhl,FURTHER_QHL=1,EXP_DATA=$experimental_data,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset -N $this_qmd_name -l $pbs_config -o $this_output_file -e $this_error_file run_qmd_instance.sh" >> $finalise_qmd_script
fi 

sleep 1

echo " 

#!/bin/bash

echo \"inside monitor script.\"
IFS=$'\n' read -d '' -r -a job_ids_s < $full_path_to_results/job_ids_started.txt
num_jobs_started=\${#job_ids_s[@]}

IFS=$'\n' read -d '' -r -a job_ids_c < $full_path_to_results/job_ids_completed.txt
num_jobs_complete=\${#job_ids_c[@]}

for k in \${job_ids_c[@]}
do
	echo \$k
done


echo \"num jobs started/finished: \$num_jobs_started \$num_jobs_complete\"

while (( \$num_jobs_complete < $num_jobs_launched ))
do
	IFS=$'\n' read -d '' -r -a job_ids_s < $full_path_to_results/job_ids_started.txt
	num_jobs_started=\${#job_ids_s[@]}

	IFS=$'\n' read -d '' -r -a job_ids_c < $full_path_to_results/job_ids_completed.txt
	num_jobs_complete=\${#job_ids_c[@]}

	echo \" jobs completed : \"
	for k in \${job_ids_c[@]}
	do
		echo \$k
	done

	echo \"Waiting. Currently \$num_jobs_complete / \$num_jobs_started \"

	sleep 3
done

sh $finalise_qmd_script
" > $monitor_script

 
chmod a+x $monitor_script
chmod a+x $finalise_qmd_script
sh $monitor_script


