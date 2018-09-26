#!/bin/bash

test_description="simulated_transverse_positive_params_only"

## Script essentials
num_tests=3
qhl=0 # do a test on QHL only -> 1; for full QMD -> 0
do_further_qhl=1 # perform further QHL parameter tuning on average values found by QMD. 
min_id=0 # update so instances don't clash and hit eachother's redis databases

# QHL parameters
p=18 # particles
e=6 # experiments
ra=0.8 #resample a 
rt=0.5 # resample threshold
rp=0.1 # PGH factor
op='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
#op='xTiPPyTiPPzTiPPxTxPPyTyPPzTzPPxTyPPxTzPPyTz'

# QMD settings
experimental_data=0 # use experimental data -> 1; use fake data ->0
#dataset='NVB_HahnPeaks_Newdata'
#dataset='NV05_HahnEcho02'
dataset='NVB_dataset.p'
#dataset='NV05_dataset.p'
data_max_time=5000 # nanoseconds
data_time_offset=205 # nanoseconds
top_number_models=2 # how many models to perform further QHL for
further_qhl_resource_factor=2
#growth_rule='two_qubit_ising_rotation_hyperfine'
growth_rule='two_qubit_ising_rotation_hyperfine_transverse'

do_plots=1
pickle_class=0
custom_prior=1
true_hamiltonian='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
if (( "$qhl" == 1 ))
then	
	rp=1.0
	num_proc=1
else
	num_proc=5
fi

if (( "$experimental_data" == 0 ))
then	
	rp=1.0
fi


### ---------------------------------------------------###
# Everything from here downwards uses the parameters
# defined above to run QMD. 
### ---------------------------------------------------###
let max_id="$min_id + $num_tests - 1 "
this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)

script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
results_dir=$day_time
full_path_to_results=$(pwd)/Results/$results_dir
all_qmd_bayes_csv="$full_path_to_results/multiQMD.csv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$full_path_to_results/$true_expec_filename"


OUT_LOG="$full_path_to_results/output_and_error_logs/"
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
max_seconds_reqd=0

echo "" > $full_path_to_results/job_ids_started.txt
echo "" > $full_path_to_results/job_ids_completed.txt

declare -a qhl_operators=(
$op
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

if (( "$qhl" == 1 ))
then
	ppn=1
else
	ppn=$num_proc
fi

node_req="nodes=1:ppn=$ppn"

printf "$day_time: \t $test_description \t e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp \n" >> QMD_Results_directories.log
num_jobs_launched=0

prior_pickle_file="$full_path_to_results/prior.p"
true_params_pickle_file="$full_path_to_results/true_params.p"

python3 ../Libraries/QML_lib/SetQHLParams.py \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -op=$op \
    -exp=$experimental_data \
    -rand_t=1 -rand_p=0 # can make true params and prior random



for op in "${qhl_operators[@]}";
do
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
							let expected_time="$ham_exp/20"
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
							if (( $seconds_reqd > $max_seconds_reqd ))
							then
								max_seconds_reqd=$seconds_reqd
							fi

							time="walltime=00:00:$seconds_reqd"
							this_qmd_name="$test_description""_$qmd_id"
							this_error_file="$OUT_LOG/$error_file""_$qmd_id.txt"
							this_output_file="$OUT_LOG/$output_file""_$qmd_id.txt"
							printf "$day_time: \t e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd \n" >> QMD_all_tasks.log

							qsub -v QMD_ID=$qmd_id,OP="$op",QHL=$qhl,FURTHER_QHL=0,EXP_DATA=$experimental_data,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset,GROWTH=$growth_rule,TRUE_PARAMS_FILE=$true_params_pickle_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$true_expec_path -N $this_qmd_name -l $node_req,$time -o $this_output_file -e $this_error_file run_qmd_instance.sh

						done
					done
				done
			done
		done
	done
done
finalise_qmd_script=$full_path_to_results/FINALISE_$test_description.sh
monitor_script=$full_path_to_results/monitor.sh
finalise_further_qhl_stage_script=$full_path_to_results/FURTHER_finalise.sh

### Generate script to analyse results of QMD runs. 
echo "
#!/bin/bash 
cd $lib_dir
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv -top=$top_number_models -qhl=$qhl -fqhl=0 -data=$dataset -exp=$experimental_data -params=$true_params_pickle_file -true_expec=$true_expec_path
" > $finalise_qmd_script

### Further QHL on best performing models. Add section to analysis script, which launches futher_qhl stage.
if (( "$do_further_qhl" == 1 ))
then
	let p="$further_qhl_resource_factor*$p"
	let e="$further_qhl_resource_factor*$e"
	let bt="$e-1"
	pgh=0.3 # further QHL on different times than initially trained on. 
	rp=2.0

	pbs_config=walltime=10:00:00,nodes=1:ppn=$top_number_models

	echo "
		qmd_id=$qmd_id
		cd $(pwd)
		for i in \`seq $min_id $max_id\`;
		do
			let qmd_id="1+\$qmd_id"
			qsub -v QMD_ID=\$qmd_id,OP="$op",QHL=0,FURTHER_QHL=1,EXP_DATA=$experimental_data,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset,GROWTH=$growth_rule,TRUE_PARAMS_FILE=$true_params_pickle_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$true_expec_path -N finalise_$test_description -l $pbs_config -o $OUT_LOG/finalise_output.txt -e $OUT_LOG/finalise_error.txt run_qmd_instance.sh 
		done 
	" >> $finalise_qmd_script

	echo "
		#!/bin/bash 
		cd $lib_dir
		python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv -top=$top_number_models -qhl=$qhl -fqhl=1 -data=$dataset -exp=$experimental_data -params=$true_params_pickle_file -true_expec=$true_expec_path
	" > $finalise_further_qhl_stage_script
	chmod a+x $finalise_further_qhl_stage_script
fi 


### Generate script to monitor instances of QMD and launch futher analysis when all instances have finished.
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

	echo \"Waiting. Currently \$num_jobs_complete / \$num_jobs_started \"
	sleep 3
done

sh $finalise_qmd_script
" > $monitor_script

 
chmod a+x $monitor_script
chmod a+x $finalise_qmd_script
let max_seconds_reqd="$max_seconds_reqd + 15"
qsub -l walltime=00:00:$max_seconds_reqd,nodes=1:ppn=1 -N monitor_$test_description -o $OUT_LOG/monitor_output.txt -e $OUT_LOG/monitor_error.txt $monitor_script


