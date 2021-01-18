#!/bin/bash

##### --------------------------------------------------------------- #####
# QMLA run configuration
##### --------------------------------------------------------------- #####
num_instances=2 # number of instances in run
run_qhl=0 # perform QHL on known (true) model
run_qhl_multi_model=0 # perform QHL for defined list of models
experiments=2 # number of experiments
particles=10 # number of particles
plot_level=5


##### --------------------------------------------------------------- #####
# Choose an exploration strategy 
# This will determine how QMLA proceeds. 
##### --------------------------------------------------------------- #####
exploration_strategy="ExampleBasic"


##### --------------------------------------------------------------- #####
# QMLA settings - default
##### --------------------------------------------------------------- #####
debug_mode=0
q_id=0 # instance ID of first instance
use_rq=0 # use RQ workers for learning
figure_format="pdf" # analysis plots stored as this type
top_number_models=5
plots=0
pickle_instances=1


##### --------------------------------------------------------------- #####
# Alternative exploration strategies,
# used if alt_exploration_strategies is not empty
##### --------------------------------------------------------------- #####
alt_exploration_strategies=(
    # 'IsingLatticeSet'
    # 'HeisenbergLatticeSet'
    # 'HubbardReducedLatticeSet'
    # 'GeneticTest'
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
do_further_qhl=0 # QHL refinement to best performing models 
further_qhl_factor=1
further_qhl_num_runs=$num_instances

# Generate all plots in QHL mode
if (( "$run_qhl" == 1 )) || (( "$run_qhl_multi_model" == 1 ))
then	
    plot_level=5
fi

# Make directories for outputs
running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
this_run_directory="$running_dir/results/$day_time/"
mkdir -p $this_run_directory
bayes_csv="$this_run_directory/all_models_bayes_factors.csv"
system_measurements_file="$this_run_directory/system_measurements.p"
run_info_file="$this_run_directory/run_info.p"
plot_probe_file="$this_run_directory/plot_probes.p"
latex_mapping_file="$this_run_directory/latex_mapping.txt"
analysis_script="$this_run_directory/analyse.sh"
log_for_entire_run="$this_run_directory/qmla.log"
further_qhl_log="$this_run_directory/qhl_further.log"
cp $(pwd)/local_launch.sh "$this_run_directory/launched_script.txt"
git_commit=$(git rev-parse HEAD)


##### --------------------------------------------------------------- #####
# First set up parameters/data to be used by all 
# instances of QMLA for this run. 
##### --------------------------------------------------------------- #####
python3 ../scripts/set_qmla_params.py \
    -dir=$this_run_directory \
    -es=$exploration_strategy \
    -prt=$particles \
    -runinfo=$run_info_file \
    -sysmeas=$system_measurements_file \
    -plotprobes=$plot_probe_file \
    -log=$log_for_entire_run \
    $exploration_strategies_command 

echo "Generated run configuration."


##### --------------------------------------------------------------- #####
# Write analysis script before launch in case
# run is stopped before some instances complete.
##### --------------------------------------------------------------- #####

echo "
cd $this_run_directory
python3 ../../../../scripts/analyse_qmla.py \
    -dir=$this_run_directory \
    --bayes_csv=$bayes_csv \
    -log=$log_for_entire_run \
    -top=$top_number_models \
    -qhl=$run_qhl \
    -fqhl=0 \
    -runinfo=$run_info_file \
    -sysmeas=$system_measurements_file \
    -es=$exploration_strategy \
    -plotprobes=$plot_probe_file \
    -latex=$latex_mapping_file \
    -gs=1 \
    -ff=$figure_format

python3 ../../../../scripts/generate_results_pdf.py \
    -t=$num_instances \
    -dir=$this_run_directory \
    -p=$particles \
    -e=$experiments\
    -log=$log_for_entire_run \
    -es=$exploration_strategy \
    -run_desc=\"localdevelopemt\" \
    -git_commit=$git_commit \
    -qhl=$run_qhl \
    -mqhl=$run_qhl_multi_model \
    -cb=$bayes_csv \

" > $analysis_script

chmod a+x $analysis_script


##### --------------------------------------------------------------- #####
# Run instances
##### --------------------------------------------------------------- #####
let max_qmla_id="$num_instances + $q_id"
for i in `seq 1 $max_qmla_id`;
do
    redis-cli flushall
    let q_id="$q_id+1"
    # python3 -m cProfile -s time \
    python3 \
        ../scripts/implement_qmla.py \
        -qid=$q_id \
        -qhl=$run_qhl \
        -mqhl=$run_qhl_multi_model \
        -p=$particles \
        -e=$experiments\
        -rq=$use_rq \
        -pl=$plot_level \
        -debug=$debug_mode \
        -dir=$this_run_directory \
        -pkl=$pickle_instances \
        -log=$log_for_entire_run \
        -cb=$bayes_csv \
        -runinfo=$run_info_file \
        -sysmeas=$system_measurements_file \
        -plotprobes=$plot_probe_file \
        -latex=$latex_mapping_file \
        -ff=$figure_format \
        -es=$exploration_strategy \
        $exploration_strategies_command \
        > $this_run_directory/output.txt
done

echo "
------ QMLA run completed ------
"


##### --------------------------------------------------------------- #####
# Further QHL (optional)
##### --------------------------------------------------------------- #####

if (( $do_further_qhl == 1 )) 
then
    sh $analysis_script

    further_analyse_filename='analyse_further_qhl.sh'
    further_analysis_script="$this_run_directory$further_analyse_filename"
    let particles="$further_qhl_factor * $particles"
    let experiments="$further_qhl_factor * $experiments"
    echo "------ Launching further QHL instance(s) ------"
    let max_qmla_id="$num_instances + 1"

    # write to a script so we can recall analysis later.
    cd $this_run_directory
    cd ../../../

    for i in \`seq 1 $max_qmla_id\`;
        do
        redis-cli flushall 
        let q_id="$q_id + 1"
        echo "QID: $q_id"
        python3 /scripts/implement_qmla.py \
            -dir=$this_run_directory \
            -fq=1 \
            -p=$particles \
            -e=$experiments \
            -rq=$use_rq \
            -qhl=0 \
            -qid=$q_id \
            -pt=$plots \
            -pkl=1 \
            -log=$log_for_entire_run \
            -cb=$bayes_csv \
            -runinfo=$run_info_file \
            -system_measurements_file=$system_measurements_file \
            -plotprobes=$plot_probe_file \
            -latex=$latex_mapping_file \
            -es=$exploration_strategy \
            $exploration_strategies_command 
    done
    echo "
    cd $this_run_directory
    python3 ../../../../scripts/AnalyseMultipleQMD.py \
        -dir=$this_run_directory \
        --bayes_csv=$bayes_csv \
        -log=$log_for_entire_run \
        -top=$top_number_models \
        -qhl=0 \
        -fqhl=1 \
        -true_expec=$system_measurements_file \
        -es=$exploration_strategy \
        -plot_probes=$plot_probe_file \
        -params=$run_info_file \
        -latex=$latex_mapping_file
    " > $further_analysis_script

    chmod a+x $further_analysis_script
    echo "------ Launching analyse further QHL ------"
    # sh $further_analysis_script
fi