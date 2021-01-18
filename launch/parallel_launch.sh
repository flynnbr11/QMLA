#!/bin/bash
run_description='example-ES__qmla-isntance'

##### --------------------------------------------------------------- #####
# QMLA run configuration
##### --------------------------------------------------------------- #####
num_instances=10 # number of instances in run
run_qhl=0 # perform QHL on known (true) model
run_qhl_multi_model=0 # perform QHL for defined list of models
experiments=500
particles=2000 
plot_level=5

##### --------------------------------------------------------------- #####
# Choose an exploration strategy 
# This will determine how QMLA proceeds. 
##### --------------------------------------------------------------- #####
exploration_strategy='ExampleBasic'


##### --------------------------------------------------------------- #####
# Timing options to enforce, in case QMLA requests too much/little time.
##### --------------------------------------------------------------- #####
time_request_insurance_factor=1
min_time_to_request=3600 # one hour by default


##### --------------------------------------------------------------- #####
# The below parameters are used by QMLA but are less crucial
# They may be changed by users as desired. 
##### --------------------------------------------------------------- #####
debug_mode=0
figure_format="pdf" # analysis plots stored as this type
top_number_models=4
do_plots=0
pickle_instances=0


##### --------------------------------------------------------------- #####
# Alternative exploration strategies,
# used only if multiple_exploration_strategies=1
##### --------------------------------------------------------------- #####
multiple_exploration_strategies=0
alt_exploration_strategies=(  
 	'IsingLatticeSet'
 	'HubbardReducedLatticeSet'
#	'HeisenbergLatticeSet'
)
exploration_strategies_command=""
for item in ${alt_exploration_strategies[*]}
do
    exploration_strategies_command+=" -aes $item" 
done


##### --------------------------------------------------------------- #####
# Everything from here downwards uses the parameters
# defined above to run QMLA. 
# These should not need to be considered by users for each 
# run, provided the default outputs are okay.
##### --------------------------------------------------------------- #####

# Further QHL on top models to refine parameters
do_further_qhl=0
further_qhl_resource_factor=1

# Create output files/directories
running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
this_run_directory=$(pwd)/results/$day_time/
qmla_dir="${running_dir%/launch}" # directory where qmla source can be found
lib_dir="$qmla_dir/qmla"
script_dir="$qmla_dir/scripts"
output_dir="$this_run_directory/output_and_error_logs/"
mkdir -p $this_run_directory
mkdir -p $output_dir

# File paths used
bayes_csv="$this_run_directory/all_models_bayes_factors.csv"
system_measurements_file="$this_run_directory/system_measurements.p"
run_info_file="$this_run_directory/run_info.p"
plot_probe_file="$this_run_directory/plot_probes.p"
latex_mapping_file="$this_run_directory/LatexMapping.txt"
git_commit="$(git rev-parse HEAD)"

cp $(pwd)/parallel_launch.sh "$this_run_directory/launched_script.txt"
echo "" > $this_run_directory/job_ids_started.txt
echo "" > $this_run_directory/job_ids_completed.txt

# Compute time to request from job scheduler
time_required_script="$this_run_directory/set_time_env_vars.sh"
touch $time_required_script
chmod a+x $time_required_script
qmla_env_var="QMLA_TIME"
qhl_env_var="QHL_TIME"
num_jobs_launched=0
log_for_entire_run="$this_run_directory/qmla_log.log"


##### --------------------------------------------------------------- #####
# Set up parameters/data to be used by all instances
# of QMLA within this run. 
##### --------------------------------------------------------------- #####
python3 ../scripts/set_qmla_params.py \
	-dir=$this_run_directory \
	-es=$exploration_strategy \
	-prt=$particles\
	-runinfo=$run_info_file \
	-sysmeas=$system_measurements_file \
	-plotprobes=$plot_probe_file \
	-log=$log_for_entire_run \
	$exploration_strategies_command

# Call script to determine how much time is needed
let temp_bayes_times="2*$experiments" # TODO fix time calculator
python3 ../scripts/time_required_calculation.py \
	-es=$exploration_strategy \
	-use_aes=$multiple_exploration_strategies \
	$exploration_strategies_command \
	-e=$experiments\
	-p=$particles\
	-proc=1 \
	-scr=$time_required_script \
	-time_insurance=$time_request_insurance_factor \
	-qmdtenv="QMLA_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME" \
	-num_proc_env="NUM_PROCESSES" \
	-mintime=$min_time_to_request

source $time_required_script
qmla_time=$QMLA_TIME
qhl_time=$QHL_TIME
fqhl_time=$FQHL_TIME
num_processes=$NUM_PROCESSES

# Change requested time. e.g. if running QHL , don't need as many nodes. 
if (( "$run_qhl" == 1 )) 
then	
	num_proc=2
	plot_level=6 # do all plots for QHL test
elif (( "run_qhl_multi_model"  == 1 ))
then 
	num_proc=5
elif (( "$multiple_exploration_strategies" == 1))
then 
	num_proc=16
else
	num_proc=$num_processes
fi

node_req="nodes=1:ppn=$num_proc"

if [ "$run_qhl" == 1 ] || [ "$run_qhl_multi_model" == 1 ] 
then
	let seconds_reqd="$qhl_time"
else
	let seconds_reqd="$qmla_time"
fi
time="walltime=00:00:$seconds_reqd"
echo "After calling scipt(s), num processes=$num_proc, seconds_reqd=$seconds_reqd"


##### --------------------------------------------------------------- #####
# Submit instances as jobs to job scheduler.
##### --------------------------------------------------------------- #####
printf "$day_time: e=$experiments; p=$particles\t $exploration_strategy \t $run_description \n" >> paths_to_results.log

min_id=0
let max_qmla_id="$min_id + $num_instances - 1 "
qmla_id=$min_id
for i in `seq $min_id $max_qmla_id`;

do
	let qmla_id="$qmla_id+1"
	let num_jobs_launched="$num_jobs_launched+1"
	this_qmla_name="$run_description""_$qmla_id"
	this_error_file="$output_dir/error_$qmla_id.txt"
	this_output_file="$output_dir/output_$qmla_id.txt"

	qsub -v RUNNING_DIR=$running_dir,LIBRARY_DIR=$lib_dir,SCRIPT_DIR=$script_dir,ROOT_DIR=$qmla_dir,QMLA_ID=$qmla_id,RUN_QHL=$run_qhl,RUN_QHL_MULTI_MODEL=$run_qhl_multi_model,FIGURE_FORMAT=$figure_format,FURTHER_QHL=0,RESULTS_DIR=$this_run_directory,DATETIME=$day_time,NUM_PARTICLES=$particles,NUM_EXPERIMENTS=$experiments,PICKLE_INSTANCE=$pickle_instances,BAYES_CSV=$bayes_csv,EXPLORATION_STRATEGY=$exploration_strategy,MULTIPLE_EXPLORATION_STRATEGIES=$multiple_exploration_strategies,ALT_ES="$exploration_strategies_command",LATEX_MAP_FILE=$latex_mapping_file,RUN_INFO_FILE=$run_info_file,SYS_MEAS_FILE=$system_measurements_file,PLOT_PROBES_FILE=$plot_probe_file,PLOT_LEVEL=$plot_level,DEBUG=$debug_mode -N $this_qmla_name -l $node_req,$time -o $this_output_file -e $this_error_file run_single_qmla_instance.sh

done
echo "Launched $num_instances instances."

finalise_qmla_script=$this_run_directory/analyse.sh
monitor_script=$this_run_directory/monitor.sh
finalise_further_qhl_stage_script=$this_run_directory/further_analyse.sh

# Generate script to analyse results of QMD runs. 
echo "
#!/bin/bash 
cd $script_dir
python3 analyse_qmla.py \
	-dir="$this_run_directory" \
	-log=$log_for_entire_run \
	--bayes_csv=$bayes_csv \
	-top=$top_number_models \
	-qhl=$run_qhl \
	-fqhl=0 \
	-plotprobes=$plot_probe_file \
	-runinfo=$run_info_file \
	-sysmeas=$system_measurements_file \
	-latex=$latex_mapping_file \
	-gs=1 \
	-ff=$figure_format \
	-es=$exploration_strategy

python3 generate_results_pdf.py \
	-dir=$this_run_directory \
	-p=$particles\
	-e=$experiments\
	-t=$num_instances \
	-es=$exploration_strategy \
	-run_desc=$run_description \
	-git_commit=$git_commit \
	-qhl=$run_qhl \
	-mqhl=$run_qhl_multi_model \
	-cb=$bayes_csv \

" > $finalise_qmla_script

# Further QHL on best performing models. Add section to analysis script, which launches futher_qhl stage.
let particles="$further_qhl_resource_factor*$particles"
let experiments="$further_qhl_resource_factor*$experiments"
pgh=1.0 # further QHL on different times than initially trained on. 
pbs_config=walltime=00:00:$fqhl_time,nodes=1:ppn=$top_number_models

# Prepare script to run further QHL if desired
# TODO out of date - match with current standards
echo "
do_further_qhl=$do_further_qhl
qmla_id=$qmla_id
cd $(pwd)
if (( "\$do_further_qhl" == 1 ))
then

	for i in \`seq $min_id $max_qmla_id\`;
	do
		let qmla_id="1+\$qmla_id"
		finalise_script="finalise_$run_description_\$qmla_id"
		qsub -v qmla_ID=\$qmla_id,QHL=0,FURTHER_QHL=1,EXP_DATA=$experimental_data,RUNNING_DIR=$running_dir,LIBRARY_DIR=$lib_dir,SCRIPT_DIR=$script_dir,RESULTS_DIR=$this_run_directory,DATETIME=$day_time,NUM_PARTICLES=$particles,NUM_EXP=$experiments,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PICKLE_qmla=$pickle_instances,BAYES_CSV=$bayes_csv,CUSTOM_PRIOR=$custom_prior,DATA_MAX_TIME=$data_max_time,EXPLORATION_STRATEGY=$exploration_strategy,LATEX_MAP_FILE=$latex_mapping_file,TRUE_PARAMS_FILE=$run_info_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$system_measurements_file,PLOT_PROBES=$plot_probe_file,RESOURCE_REALLOCATION=$resource_reallocation,UPDATER_FROM_PRIOR=$updater_from_prior,GAUSSIAN=$gaussian,PARAM_MIN=$param_min,PARAM_MAX=$param_max,PARAM_MEAN=$param_mean,PARAM_SIGMA=$param_sigma -N \$finalise_script -l $pbs_config -o $output_dir/finalise_output.txt -e $output_dir/finalise_error.txt run_qmla_instance.sh 
	done 
fi
" >> $finalise_qmla_script

echo "
	#!/bin/bash 
	cd $script_dir
	python3 analyse_qmla.py \
		-dir="$this_run_directory" \
		-log=$log_for_entire_run \
		--bayes_csv=$bayes_csv \
		-top=$top_number_models 
		-qhl=$run_qhl \
		-fqhl=1 \
		-latex==$latex_mapping_file \
		-runinfo=$run_info_file \
		-sysmeas=$system_measurements_file \
		-es=$exploration_strategy \
		-plotprobes=$plot_probe_file

	python3 generate_results_pdf.py \
		-dir=$this_run_directory \
		-p=$particles \
		-e=$experiments \
		-t=$num_instances \
		-nprobes=$num_probes \
		-pnoise=$probe_noise_level \
		-special_probe=$special_probe \
		-es=$exploration_strategy \
		-run_desc=$run_description \
		-git_commit=$git_commit \
		-ra=$ra \
		-rt=$rt \
		-pgh=$rp \
		-qhl=$run_qhl \
		-mqhl=$run_qhl_multi_model \
		-cb=$bayes_csv \
		-exp=$experimental_data \
		-out="further_qhl_analysis"
" > $finalise_further_qhl_stage_script
chmod a+x $finalise_further_qhl_stage_script

# Generate script to monitor instances and launch futher analysis when all finished.
echo " 
#!/bin/bash

echo \"inside monitor script.\"
IFS=$'\n' read -d '' -r -a job_ids_s < $this_run_directory/job_ids_started.txt
num_jobs_started=\${#job_ids_s[@]}

IFS=$'\n' read -d '' -r -a job_ids_c < $this_run_directory/job_ids_completed.txt
num_jobs_complete=\${#job_ids_c[@]}

for k in \${job_ids_c[@]}
do
	echo \$k
done


echo \"num jobs started/finished: \$num_jobs_started \$num_jobs_complete\"

while (( \$num_jobs_complete < $num_jobs_launched ))
do
	IFS=$'\n' read -d '' -r -a job_ids_s < $this_run_directory/job_ids_started.txt
	num_jobs_started=\${#job_ids_s[@]}

	IFS=$'\n' read -d '' -r -a job_ids_c < $this_run_directory/job_ids_completed.txt
	num_jobs_complete=\${#job_ids_c[@]}

	echo \"Waiting. Currently \$num_jobs_complete / \$num_jobs_started \"
	sleep 3
done

sh $finalise_qmla_script
" > $monitor_script

let max_seconds_reqd="$seconds_reqd + 15"
chmod a+x $monitor_script
chmod a+x $finalise_qmla_script
#qsub -l walltime=00:00:$max_seconds_reqd,nodes=1:ppn=1 -N monitor_$run_description -o $output_dir/monitor_output.txt -e $output_dir/monitor_error.txt $monitor_script
