#!/bin/bash
# note monitor script currently turned off (at very bottom)
# test_description='NV-GA__prelearned-models__6-qubit-space__4-qubit-true'
# test_description='theory-paper__selection-from-lattices__heis-rerun'
# test_description='multiple-GR-at-once__true-ising__DEBUG'
# test_description='ga__4-qubit-HeisXYZ__IQLE__Elo__28-models-32-gen__reassigning-fitness-at-rebirth'
# test_description='time-tests__fh-4-sites'
# test_description='lattices_fh__4-site-square__qhl'
test_description='test-hexp-install__fh-qhl'


### ---------------------------------------------------###
# Essential choices for how to run multiple 
# instances of QMD from this script. 
### ---------------------------------------------------###

## Number and type of QMLA instances to perform in this run.
num_instances=5
run_qhl=1 # do a test on QHL only -> 1; for full QMD -> 0
run_qhl_multi_model=0
multiple_growth_rules=0
do_further_qhl=0 # perform further QHL parameter tuning on average values found by QMLA. 
plot_level=2
debug_mode=0
time_request_insurance_factor=1
min_time_to_request=16000 # 1300 by default

# QHL parameters.
e=100 # experiments
p=2000 # particles

### ---------------------------------------------------###
# Choose growth rule 
# will be determined by sim_rowth_rule, exp_growth_rule, 
# and value of experimental_data.
### ---------------------------------------------------###

# growth_rule='NVCentreGenticAlgorithmPrelearnedParameters'
# growth_rule='NVCentreSimulatedLongDynamicsGenticAlgorithm'
# growth_rule='FermiHubbardLatticeSet'
# growth_rule='NVCentreNQubitBath'
# growth_rule='HeisenbergGeneticXXZ'
# growth_rule='IsingGeneticTest'
# growth_rule='TestSimulatedNVCentre'
# growth_rule='IsingGeneticSingleLayer'
# growth_rule='ObjFncBFP'
# growth_rule='IsingGenetic'
# growth_rule='HeisenbergGeneticXXZ'

# growth_rule='NVCentreSimulatedShortDynamicsGenticAlgorithm'
# growth_rule='NVCentreExperimentalShortDynamicsGenticAlgorithm'
# gowth_rule='NVCentreRevivals'
# growth_rule='NVCentreRevivalsSimulated'
# growth_rule='NVCentreRevivalSimulation'
# growth_rule='IsingGenetic'
# growth_rule='ExperimentNVCentreNQubits'
# growth_rule='SimulatedNVCentre'

# growth_rule='FermiHubbardLatticeSet'
growth_rule='IsingLatticeSet'
# growth_rule='HeisenbergLatticeSet'

# Alternative growth rules, i.e. to learn alongside the true one. Used if multiple_growth_rules set to 1 above
alt_growth_rules=(  
	'IsingLatticeSet'
# 	'FermiHubbardLatticeSet'
	'HeisenbergLatticeSet'
)
growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done


### ---------------------------------------------------###
# The below parameters are used by QMD. 
# These should be considered by the user to ensure they match requirements. 
### ---------------------------------------------------###

# QMD settings - for learning (QHL) and comparison (BF)
further_qhl_resource_factor=1
do_plots=0
pickle_class=0
top_number_models=4

### ---------------------------------------------------###
# Everything from here downwards uses the parameters
# defined above to run QMD. 
# These do not need to be considered for every instance of QMD provided the default outputs are okay.
### ---------------------------------------------------###

# Create output files/directories
running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
this_run_directory=$(pwd)/Results/$day_time/
qmla_dir="${running_dir%/launch}" # directory where qmla source can be found
lib_dir="$qmla_dir/qmla"
script_dir="$qmla_dir/scripts"
output_dir="$this_run_directory/output_and_error_logs/"
mkdir -p $this_run_directory
mkdir -p $output_dir

# results_dir=$day_time

# File paths used
bayes_csv="$this_run_directory/all_models_bayes_factors.csv"
system_measurements_file="$this_run_directory/system_measurements.p"
run_info_file="$this_run_directory/true_params.p"
plot_probe_file="$this_run_directory/plot_probes.p"
latex_mapping_file="$this_run_directory/LatexMapping.txt"



cp $(pwd)/parallel_launch.sh "$this_run_directory/launched_script.txt"
global_server=$(hostname) # TODO can be removed ?
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
git_commit="$(git rev-parse HEAD)"

### ---------------------------------------------------###
# Lists of QMD/QHL/BF params to loop over, e.g for parameter sweep
# Default is to only include parameters as defined above. 
### ---------------------------------------------------###

### First set up parameters/data to be used by all instances of QMD for this run. 
# python3 ../qmla/SetQHLParams.py \
python3 ../scripts/set_qmla_params.py \
	-dir=$this_run_directory \
	-ggr=$growth_rule \
	-prt=$p \
	-runinfo=$run_info_file \
	-sysmeas=$system_measurements_file \
	-plotprobes=$plot_probe_file \
	-log=$log_for_entire_run \
	$growth_rules_command

### Call script to determine how much time is needed based on above params. Store in qmla_TIME, QHL_TIME, etc. 
let temp_bayes_times="2*$e" # TODO fix time calculator
python3 ../scripts/time_required_calculation.py \
	-ggr=$growth_rule \
	-use_agr=$multiple_growth_rules \
	$growth_rules_command \
	-e=$e \
	-p=$p \
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
num_processes=$NUM_PROCESSES # TODO RESTORE!!!!!!! testing without RQ

# Change requested time. e.g. if running QHL , don't need as many nodes. 
if (( "$run_qhl" == 1 )) 
then	
	num_proc=2
	plot_level=6 # do all plots for QHL test
elif (( "run_qhl_multi_model"  == 1 ))
then 
	num_proc=5
elif (( "$multiple_growth_rules" == 1))
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


### ---------------------------------------------------###
# Submit instances as jobs to job scheduler.
### ---------------------------------------------------###
printf "$day_time: e=$e; p=$p \t $growth_rule \t $test_description \n" >> paths_to_results.log

min_id=0
let max_id="$min_id + $num_instances - 1 "
qmla_id=$min_id
for i in `seq $min_id $max_id`;

do
	let qmla_id="$qmla_id+1"
	let num_jobs_launched="$num_jobs_launched+1"
	this_qmla_name="$test_description""_$qmla_id"
	this_error_file="$output_dir/error_$qmla_id.txt"
	this_output_file="$output_dir/output_$qmla_id.txt"

	qsub -v RUNNING_DIR=$running_dir,LIBRARY_DIR=$lib_dir,SCRIPT_DIR=$script_dir,ROOT_DIR=$qmla_dir,QMLA_ID=$qmla_id,RUN_QHL=$run_qhl,RUN_QHL_MULTI_MODEL=$run_qhl_multi_model,FURTHER_QHL=0,GLOBAL_SERVER=$global_server,RESULTS_DIR=$this_run_directory,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXPERIMENTS=$e,PLOTS=$do_plots,PICKLE_INSTANCE=$pickle_class,BAYES_CSV=$bayes_csv,GROWTH_RULE=$growth_rule,MULTIPLE_GROWTH_RULES=$multiple_growth_rules,ALT_GROWTH="$growth_rules_command",LATEX_MAP_FILE=$latex_mapping_file,RUN_INFO_FILE=$run_info_file,SYS_MEAS_FILE=$system_measurements_file,PLOT_PROBES_FILE=$plot_probe_file,PLOT_LEVEL=$plot_level,DEBUG=$debug_mode -N $this_qmla_name -l $node_req,$time -o $this_output_file -e $this_error_file run_single_qmla_instance.sh

done

finalise_qmla_script=$this_run_directory/FINALISE_$test_description.sh
monitor_script=$this_run_directory/monitor.sh
finalise_further_qhl_stage_script=$this_run_directory/FURTHER_finalise.sh

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
	-ggr=$growth_rule

python3 generate_results_pdf.py \
	-dir=$this_run_directory \
	-p=$p \
	-e=$e \
	-t=$num_instances \
	-ggr=$growth_rule \
	-run_desc=$test_description \
	-git_commit=$git_commit \
	-qhl=$run_qhl \
	-mqhl=$run_qhl_multi_model \
	-cb=$bayes_csv \

" > $finalise_qmla_script


# Further QHL on best performing models. Add section to analysis script, which launches futher_qhl stage.
let p="$further_qhl_resource_factor*$p"
let e="$further_qhl_resource_factor*$e"
let bt="$e-1"
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

	for i in \`seq $min_id $max_id\`;
	do
		let qmla_id="1+\$qmla_id"
		finalise_script="finalise_$test_description_\$qmla_id"
		qsub -v qmla_ID=\$qmla_id,QHL=0,FURTHER_QHL=1,EXP_DATA=$experimental_data,RUNNING_DIR=$running_dir,LIBRARY_DIR=$lib_dir,SCRIPT_DIR=$script_dir,GLOBAL_SERVER=$global_server,RESULTS_DIR=$this_run_directory,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_qmla=$pickle_class,BAYES_CSV=$bayes_csv,CUSTOM_PRIOR=$custom_prior,DATA_MAX_TIME=$data_max_time,GROWTH=$growth_rule,LATEX_MAP_FILE=$latex_mapping_file,TRUE_PARAMS_FILE=$run_info_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$system_measurements_file,PLOT_PROBES=$plot_probe_file,RESOURCE_REALLOCATION=$resource_reallocation,UPDATER_FROM_PRIOR=$updater_from_prior,GAUSSIAN=$gaussian,PARAM_MIN=$param_min,PARAM_MAX=$param_max,PARAM_MEAN=$param_mean,PARAM_SIGMA=$param_sigma -N \$finalise_script -l $pbs_config -o $output_dir/finalise_output.txt -e $output_dir/finalise_error.txt run_qmla_instance.sh 
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
		-ggr=$growth_rule \
		-plotprobes=$plot_probe_file

	python3 generate_results_pdf.py \
		-dir=$this_run_directory \
		-p=$p -e=$e -bt=$bt -t=$num_instances \
		-nprobes=$num_probes \
		-pnoise=$probe_noise_level \
		-special_probe=$special_probe \
		-ggr=$growth_rule \
		-run_desc=$test_description \
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



### Generate script to monitor instances of QMD and launch futher analysis when all instances have finished.
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
#qsub -l walltime=00:00:$max_seconds_reqd,nodes=1:ppn=1 -N monitor_$test_description -o $output_dir/monitor_output.txt -e $output_dir/monitor_error.txt $monitor_script
