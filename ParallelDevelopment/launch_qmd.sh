#!/bin/bash
# note monitor script currently turned off (at very bottom)
test_description="custom_ham_exp_new_code"

## Script essentials
num_tests=30
qhl=0 # do a test on QHL only -> 1; for full QMD -> 0
do_further_qhl=0 # perform further QHL parameter tuning on average values found by QMD. 
min_id=0 # update so instances don't clash and hit eachother's redis databases
experimental_data=1 # use experimental data -> 1; use fake data ->0
simulate_experiment=0

# QHL parameters
p=1000 # particles
e=100 # experiments
ra=0.8 #resample a 
rt=0.5 # resample threshold
rp=0.5 # PGH factor
#op='xTxTTx'
op='xTxTTiPPPiTxTTx'
#op='xTxTTiPPPiTxTTx'
#op='yTz'
#op='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
#op='xTiPPyTiPPzTiPPxTxPPyTyPPzTzPPxTyPPxTzPPyTz'

# QMD settings
#dataset='NVB_HahnPeaks_Newdata'
#dataset='NV05_HahnEcho02'
dataset='NVB_dataset.p'
#dataset='NVB_rescale_dataset.p'
#dataset='NV05_dataset.p'
sim_measurement_type='full_access'
exp_measurement_type='hahn' # to use if not experimental



data_max_time=5000 # nanoseconds
data_time_offset=205 # nanoseconds
top_number_models=3 # how many models to perform further QHL for
further_qhl_resource_factor=1
#growth_rule='two_qubit_ising_rotation_hyperfine'
growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
#growth_rule='non_interacting_ising_single_axis'
#growth_rule='non_interacting_ising'
#growth_rule='deterministic_noninteracting_ising_single_axis'
#growth_rule='interacting_nearest_neighbour_ising'
#growth_rule='interacing_nn_ising_fixed_axis'
#growth_rule='deterministic_interacting_nn_ising_single_axis'
#growth_rule='deterministic_transverse_ising_nn_fixed_axis'
#growth_rule='heisenberg_nontransverse'
#growth_rule='heisenberg_transverse'
#growth_rule='hubbard'
#growth_rule='hubbard_chain_just_hopping'
#growth_rule='hubbard_chain'
#growth_rule='hubbard_square_lattice_generalised'


multiple_growth_rules=0 # NOTE this is being manually passed to CalculateTimeRequired below #TODO make it acceot this arg
alt_growth_rules=(
	'interacing_nn_ising_fixed_axis'
	'heisenberg_transverse'
)
growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done

do_plots=0
pickle_class=0
custom_prior=1

gaussian=1 # set to 0 for uniform distribution, 1 for normal
param_min=0
param_max=8
param_mean=0.5
param_sigma=0.3
random_true_params=1 # if not random, then as set in Libraries/QML_Lib/SetQHLParams.py
random_prior=0 # if not random, then as set in Libraries/QML_Lib/SetQHLParams.py

# Overwrite settings for specific cases
if (( "$experimental_data" == 1)) # NOTE use round brackets for arithmetic comparisons in bash; square brackets for strings
then
	echo "experimental data = $experimental_data "
	measurement_type=$exp_measurement_type
	rp=1.0
	multiple_growth_rules=0
	growth_rule='two_qubit_ising_rotation_hyperfine'
#	growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
	op='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
elif [[ "$growth_rule" == 'two_qubit_ising_rotation_hyperfine' ]] 
then
	measurement_type=$exp_measurement_type
elif [[ "$growth_rule" == 'two_qubit_ising_rotation_hyperfine_transverse' ]] 
then
	measurement_type=$exp_measurement_type
else
    measurement_type=$sim_measurement_type
fi

if (( "$qhl" == 1 ))
then	
	rp=1.0
	num_proc=1
else
	num_proc=5
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
all_qmd_bayes_csv="$full_path_to_results/cumulative.csv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$full_path_to_results/$true_expec_filename"
latex_map_name='LatexMapping.txt'
latex_mapping_file=$full_path_to_results/$latex_map_name
resource_reallocation=0

copied_launch_file="$full_path_to_results/launched_script.txt"
touch $copied_launch_file
cat $(pwd)/launch_qmd.sh > $copied_launch_file
echo "copied launch script to results directory"

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
cutoff_time=600
max_seconds_reqd=0

echo "" > $full_path_to_results/job_ids_started.txt
echo "" > $full_path_to_results/job_ids_completed.txt

declare -a qhl_operators=(
$op
)

declare -a resample_a_values=(
$ra
#0.8
#0.9
#0.98
)

declare -a resample_thresh_values=(
$rt
#0.4
#0.5
#0.7
)

declare -a pgh_values=(
$rp
#0.1
#0.3
#0.5
#0.75
#1.0
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
plot_probe_file="$full_path_to_results/plot_probes.p"
force_plot_plus=0
special_probe='random' #'ideal'

if (( "$simulate_experiment" == 1))
then
#	special_probe='plus' # test simulation using plus probe only.
	special_probe='plus_random' # test simulation using plus probe only.
	growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
	measurement_type=$exp_measurement_type
 
elif (( "$experimental_data" == 1))
then
	force_plot_plus=1
	special_probe='plus_random'
fi


python3 ../Libraries/QML_lib/SetQHLParams.py \
	-true=$true_params_pickle_file \
	-prior=$prior_pickle_file \
	-probe=$plot_probe_file \
	-plus=$force_plot_plus \
	-sp=$special_probe \
	-g=$gaussian \
	-min=$param_min \
	-max=$param_max \
	-mean=$param_mean \
	-sigma=$param_sigma \
	-op=$op \
	-exp=$experimental_data \
	-ggr=$growth_rule \
	-rand_t=$random_true_params \
	-rand_p=$random_prior # can make true params and prior random

time_required_script="$full_path_to_results/set_time_env_vars.sh"
touch $time_required_script
chmod a+x $time_required_script

qmd_env_var="QMD_TIME"
qhl_env_var="QHL_TIME"
let temp_bayes_times="$p" # TODO fix time calculator

python3 ../Libraries/QML_lib/CalculateTimeRequired.py \
	-ggr=$growth_rule \
	-use_agr=$multiple_growth_rules \
	$growth_rules_command \
	-e=$e \
	-p=$p \
	-bt=$temp_bayes_times \
	-proc=1 \
	-res=$resource_reallocation \
	-scr=$time_required_script \
	-qmdtenv="QMD_TIME" \
	-qhltenv="QHL_TIME" \
	-fqhltenv="FQHL_TIME" \
	-mintime=1300

source $time_required_script
qmd_time=$QMD_TIME
qhl_time=$QHL_TIME
fqhl_time=$FQHL_TIME

echo "QMD TIME: $qmd_time"
echo "QHL TIME: $qhl_time"


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
							let bt="$e/2"
							let qmd_id="$qmd_id+1"

							if [ "$qhl" == 1 ]
							then
								let seconds_reqd="$qhl_time"
							else
								let seconds_reqd="$qmd_time"
							fi

							let num_jobs_launched="$num_jobs_launched+1"
							time="walltime=00:00:$seconds_reqd"
							this_qmd_name="$test_description""_$qmd_id"
							this_error_file="$OUT_LOG/$error_file""_$qmd_id.txt"
							this_output_file="$OUT_LOG/$output_file""_$qmd_id.txt"
							printf "$day_time: \t e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd \n" >> QMD_all_tasks.log

							qsub -v QMD_ID=$qmd_id,OP="$op",QHL=$qhl,FURTHER_QHL=0,EXP_DATA=$experimental_data,MEAS=$measurement_type,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset,GROWTH=$growth_rule,MULTIPLE_GROWTH_RULES=$multiple_growth_rules,ALT_GROWTH="$growth_rules_command",LATEX_MAP_FILE=$latex_mapping_file,TRUE_PARAMS_FILE=$true_params_pickle_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$true_expec_path,PLOT_PROBES=$plot_probe_file,SPECIAL_PROBE=$special_probe,RESOURCE_REALLOCATION=$resource_reallocation,GAUSSIAN=$gaussian,PARAM_MIN=$param_min,PARAM_MAX=$param_max,PARAM_MEAN=$param_mean,PARAM_SIGMA=$param_sigma -N $this_qmd_name -l $node_req,$time -o $this_output_file -e $this_error_file run_qmd_instance.sh

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
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv -top=$top_number_models -qhl=$qhl -fqhl=0 -data=$dataset -plot_probes=$plot_probe_file \
	-exp=$experimental_data -meas=$measurement_type -params=$true_params_pickle_file -true_expec=$true_expec_path -latex=$latex_mapping_file -ggr=$growth_rule
" > $finalise_qmd_script

### Further QHL on best performing models. Add section to analysis script, which launches futher_qhl stage.
if (( "$do_further_qhl" == 1 ))
then
	let p="$further_qhl_resource_factor*$p"
	let e="$further_qhl_resource_factor*$e"
	let bt="$e-1"
	pgh=1.0 # further QHL on different times than initially trained on. 
	rp=2.0

	pbs_config=walltime=00:00:$fqhl_time,nodes=1:ppn=$top_number_models

	echo "
qmd_id=$qmd_id
cd $(pwd)
for i in \`seq $min_id $max_id\`;
do
	let qmd_id="1+\$qmd_id"
	qsub -v QMD_ID=\$qmd_id,OP="$op",QHL=0,FURTHER_QHL=1,EXP_DATA=$experimental_data,MEAS=$measurement_type,GLOBAL_SERVER=$global_server,RESULTS_DIR=$full_path_to_results,DATETIME=$day_time,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv,CUSTOM_PRIOR=$custom_prior,DATASET=$dataset,DATA_MAX_TIME=$data_max_time,DATA_TIME_OFFSET=$data_time_offset,GROWTH=$growth_rule,LATEX_MAP_FILE=$latex_mapping_file,TRUE_PARAMS_FILE=$true_params_pickle_file,PRIOR_FILE=$prior_pickle_file,TRUE_EXPEC_PATH=$true_expec_path,PLOT_PROBES=$plot_probe_file,RESOURCE_REALLOCATION=$resource_reallocation,GAUSSIAN=$gaussian,PARAM_MIN=$param_min,PARAM_MAX=$param_max,PARAM_MEAN=$param_mean,PARAM_SIGMA=$param_sigma -N finalise_$test_description\_\$qmd_id -l $pbs_config -o $OUT_LOG/finalise_output.txt -e $OUT_LOG/finalise_error.txt run_qmd_instance.sh 
done 
	" >> $finalise_qmd_script

	echo "
		#!/bin/bash 
		cd $lib_dir
		python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv -top=$top_number_models -qhl=$qhl -fqhl=1 -data=$dataset -exp=$experimental_data -meas=$measurement_type -latex==$latex_mapping_file -params=$true_params_pickle_file -true_expec=$true_expec_path -ggr=$growth_rule -plot_probes=$plot_probe_file
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

let max_seconds_reqd="$seconds_reqd + 15"
chmod a+x $monitor_script
chmod a+x $finalise_qmd_script
#qsub -l walltime=00:00:$max_seconds_reqd,nodes=1:ppn=1 -N monitor_$test_description -o $OUT_LOG/monitor_output.txt -e $OUT_LOG/monitor_error.txt $monitor_script


