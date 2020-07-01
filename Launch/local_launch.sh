#!/bin/bash

# redis-server

###############
# QMLA run configuration
###############
num_tests=1
exp=5 # number of experiments
prt=10 # number of particles
qhl_test=0 # perform QHL on known (true) model
multiple_qhl=0 # perform QHL for defined list of models.
do_further_qhl=0 # QHL refinement to best performing models 
q_id=0 # can start from other ID if desired


###############
# QMLA settings
###############
use_rq=0
further_qhl_factor=1
further_qhl_num_runs=$num_tests
plots=0
number_best_models_further_qhl=5

###############
# Choose a growth rule This will determine how QMD proceeds. 
# use_alt_growth_rules=1 # note this is redundant locally, currently
###############

# growth_rule='TestSimulatedNVCentre'
# growth_rule='IsingGeneticTest'
# growth_rule='IsingGeneticSingleLayer'
# growth_rule='NVCentreRevivals'
# growth_rule='NVCentreRevivalsSimulated'

# growth_rule='IsingGenetic'
# growth_rule='SimulatedNVCentre'
# growth_rule='ExperimentNVCentreNQubits'
# growth_rule='NVCentreSimulatedShortDynamicsGenticAlgorithm'
# growth_rule='NVCentreExperimentalShortDynamicsGenticAlgorithm'
growth_rule='NVCentreNQubitBath'

# growth_rule='IsingLatticeSet'
# growth_rule='HeisenbergLatticeSet'
# growth_rule='FermiHubbardLatticeSet'
# growth_rule='NVLargeSpinBath'
# growth_rule='GeneticTest'
# growth_rule='Genetic'
# growth_rule='NVExperimentalData'

alt_growth_rules=(
    # 'GeneticTest'
)

growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done

###############
# Parameters from here downwards uses the parameters
# defined above to run QMLA. 
###############

let max_qmd_id="$num_tests + $q_id"

# Files where output will be stored
running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
full_path_to_results="$running_dir/Results/$day_time/"
# qmd_dir="${running_dir%/ExperimentalSimulations}"
# lib_dir="$qmd_dir/Libraries/QML_lib"
bayes_csv="$full_path_to_results/cumulative.csv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$full_path_to_results/system_measurements.p"
prior_pickle_file="$full_path_to_results/prior.p"
true_params_pickle_file="$full_path_to_results/true_params.p"
plot_probe_file="$full_path_to_results/plot_probes.p"
latex_mapping_filename='LatexMapping.txt'
latex_mapping_file=$full_path_to_results$latex_mapping_filename
analyse_filename='analyse.sh'
analyse_script="$full_path_to_results$analyse_filename"
this_log="$full_path_to_results/qmd.log"
further_qhl_log="$full_path_to_results/qhl_further.log"
mkdir -p $full_path_to_results
# Copy some files into results directory
copied_launch_file="$full_path_to_results/launched_script.txt"
cp $(pwd)/local_launch.sh $copied_launch_file
git_commit=$(git rev-parse HEAD)


###############
# First set up parameters/data to be used by all instances of QMD for this run. 
###############

python3 ../scripts/set_qmla_params.py \
    -prt=$prt \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -probe=$plot_probe_file \
    -ggr=$growth_rule \
    -dir=$full_path_to_results \
    -log=$this_log \
    -true_expec_path=$true_expec_path \
    $growth_rules_command 

echo "Generated configuration."

###############
# Write analysis script 
# before launch in case run stopped before some instances complete.
###############

echo "
cd $full_path_to_results
python3 ../../../../scripts/analyse_qmla.py \
    -dir=$full_path_to_results \
    --bayes_csv=$bayes_csv \
    -log=$this_log \
    -top=$number_best_models_further_qhl \
    -qhl=$qhl_test \
    -fqhl=0 \
    -true_expec=$true_expec_path \
    -ggr=$growth_rule \
    -plot_probes=$plot_probe_file \
    -params=$true_params_pickle_file \
    -latex=$latex_mapping_file \
    -gs=1

python3 ../../../../scripts/generate_results_pdf.py \
    -t=$num_tests \
    -dir=$full_path_to_results \
    -p=$prt \
    -e=$exp \
    -log=$this_log \
    -ggr=$growth_rule \
    -run_desc=\"localdevelopemt\" \
    -git_commit=$git_commit \
    -qhl=$qhl_test \
    -mqhl=$multiple_qhl \
    -cb=$bayes_csv \

" > $analyse_script

chmod a+x $analyse_script

###############
# Run instances
###############
for i in `seq 1 $max_qmd_id`;
do
    redis-cli flushall
    let q_id="$q_id+1"
    # python3 -m cProfile -s time \
    python3 \
        ../scripts/implement_qmla.py \
        -qhl=$qhl_test \
        -mqhl=$multiple_qhl \
        -rq=$use_rq \
        -p=$prt \
        -e=$exp \
        -qid=$q_id \
        -log=$this_log \
        -dir=$full_path_to_results \
        -pt=$plots \
        -pkl=1 \
        -cb=$bayes_csv \
        -prior_path=$prior_pickle_file \
        -true_params_path=$true_params_pickle_file \
        -true_expec_path=$true_expec_path \
        -plot_probes=$plot_probe_file \
        -latex=$latex_mapping_file \
        -ggr=$growth_rule \
        $growth_rules_command \
        > $full_path_to_results/output.txt
done

echo "
------ QMLA completed ------
"

###############
# Furhter QHL, optionally
###############

if (( $do_further_qhl == 1 )) 
then
    sh $analyse_script

    further_analyse_filename='analyse_further_qhl.sh'
    further_analyse_script="$full_path_to_results$further_analyse_filename"
    let particles="$further_qhl_factor * $prt"
    let experiments="$further_qhl_factor * $exp"
    echo "------ Launching further QHL instance(s) ------"
    let max_qmd_id="$num_tests + 1"

    # write to a script so we can recall analysis later.
    cd $full_path_to_results
    cd ../../../

    for i in \`seq 1 $max_qmd_id\`;
        do
        redis-cli flushall 
        let q_id="$q_id + 1"
        echo "QID: $q_id"
        python3 /scripts/implement_qmla.py \
            -fq=1 \
            -p=$particles \
            -e=$experiments \
            -rq=$use_rq \
            -qhl=0 \
            -dir=$full_path_to_results \
            -qid=$q_id \
            -pt=$plots \
            -pkl=1 \
            -log=$this_log \
            -cb=$bayes_csv \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -latex=$latex_mapping_file \
            -ggr=$growth_rule \
            -ggr=$growth_rule \
            $growth_rules_command 
    done
    echo "
    cd $full_path_to_results
    python3 ../../../../scripts/AnalyseMultipleQMD.py \
        -dir=$full_path_to_results \
        --bayes_csv=$bayes_csv \
        -log=$this_log \
        -top=$number_best_models_further_qhl \
        -qhl=0 \
        -fqhl=1 \
        -true_expec=$true_expec_path \
        -ggr=$growth_rule \
        -plot_probes=$plot_probe_file \
        -params=$true_params_pickle_file \
        -latex=$latex_mapping_file
    " > $further_analyse_script

    chmod a+x $further_analyse_script
    echo "------ Launching analyse further QHL ------"
    # sh $further_analyse_script
fi


# redis-cli shutdown
