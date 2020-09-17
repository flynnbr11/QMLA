#!/bin/bash

# redis-server

###############
# QMLA run configuration
###############
num_instances=1
run_qhl=1 # perform QHL on known (true) model
run_qhl_mulit_model=0 # perform QHL for defined list of models.
do_further_qhl=0 # QHL refinement to best performing models 
q_id=0 # isntance ID can start from other ID if desired
exp=500 # number of experiments
prt=2000 # number of particles

###############
# QMLA settings
###############
use_rq=0
further_qhl_factor=1
further_qhl_num_runs=$num_instances
plots=0
number_best_models_further_qhl=5
plot_level=2
debug_mode=0

###############
# Choose a growth rule 
# This will determine how QMLA proceeds. 
# use_alt_growth_rules=1 # note this is redundant locally, currently
###############

# growth_rule='TestSimulatedNVCentre'
# growth_rule='NVCentreRevivals'
# growth_rule='NVCentreRevivalsSimulated'
# growth_rule='NVCentreNQubitBath'
# growth_rule='NVCentreGenticAlgorithmPrelearnedParameters'
# growth_rule='NVPrelearnedTest'
# growth_rule='IsingLatticeSet'
# growth_rule='HeisenbergLatticeSet'
# growth_rule='FermiHubbardLatticeSet'
# growth_rule='Demonstration'

# growth_rule='HeisenbergGenetic'
growth_rule='HeisenbergGeneticXXZ'
# growth_rule='IsingGenetic'
# growth_rule='IsingGeneticTest'
# growth_rule='HeisenbergGeneticXXZ'
# growth_rule='IsingGeneticSingleLayer'
# growth_rule='GenAlgObjectiveFncTest'
# growth_rule='ObjFncResiduals'
# growth_rule='ObjFncElo'
# growth_rule='SimulatedNVCentre'
# growth_rule='ExperimentNVCentreNQubits'
# growth_rule='NVCentreSimulatedShortDynamicsGenticAlgorithm'
# growth_rule='NVCentreExperimentalShortDynamicsGenticAlgorithm'
# growth_rule='NVCentreRevivalSimulation'
# growth_rule='NVCentreSimulatedLongDynamicsGenticAlgorithm'


# growth_rule='IsingLatticeSet'
# growth_rule='HeisenbergLatticeSet'
# growth_rule='NVLargeSpinBath'
# growth_rule='GeneticTest'
# growth_rule='Genetic'
# growth_rule='NVExperimentalData'

alt_growth_rules=(
    # 'IsingLatticeSet'
    # 'HeisenbergLatticeSet'
    # 'FermiHubbardLatticeSet'
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
# e.g. to create filepaths to use during QMLA.
###############

let max_qmd_id="$num_instances + $q_id"

running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
this_run_directory="$running_dir/Results/$day_time/"
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

###############
# First set up parameters/data to be used by all instances of QMLA for this run. 
###############

python3 ../scripts/set_qmla_params.py \
    -dir=$this_run_directory \
    -ggr=$growth_rule \
    -prt=$prt \
    -runinfo=$run_info_file \
    -sysmeas=$system_measurements_file \
    -plotprobes=$plot_probe_file \
    -log=$log_for_entire_run \
    $growth_rules_command 

echo "Generated configuration."

###############
# Write analysis script 
# before launch in case run stopped before some instances complete.
###############

echo "
cd $this_run_directory
python3 ../../../../scripts/analyse_qmla.py \
    -dir=$this_run_directory \
    --bayes_csv=$bayes_csv \
    -log=$log_for_entire_run \
    -top=$number_best_models_further_qhl \
    -qhl=$run_qhl \
    -fqhl=0 \
    -runinfo=$run_info_file \
    -sysmeas=$system_measurements_file \
    -ggr=$growth_rule \
    -plotprobes=$plot_probe_file \
    -latex=$latex_mapping_file \
    -gs=1

python3 ../../../../scripts/generate_results_pdf.py \
    -t=$num_instances \
    -dir=$this_run_directory \
    -p=$prt \
    -e=$exp \
    -log=$log_for_entire_run \
    -ggr=$growth_rule \
    -run_desc=\"localdevelopemt\" \
    -git_commit=$git_commit \
    -qhl=$run_qhl \
    -mqhl=$run_qhl_mulit_model \
    -cb=$bayes_csv \

" > $analysis_script

chmod a+x $analysis_script

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
        -qid=$q_id \
        -qhl=$run_qhl \
        -mqhl=$run_qhl_mulit_model \
        -p=$prt \
        -e=$exp \
        -rq=$use_rq \
        -pt=$plots \
        -pl=$plot_level \
        -debug=$debug_mode \
        -dir=$this_run_directory \
        -pkl=1 \
        -log=$log_for_entire_run \
        -cb=$bayes_csv \
        -runinfo=$run_info_file \
        -sysmeas=$system_measurements_file \
        -plotprobes=$plot_probe_file \
        -latex=$latex_mapping_file \
        -ggr=$growth_rule \
        $growth_rules_command \
        > $this_run_directory/output.txt
done


echo "
------ QMLA completed ------
"

###############
# Furhter QHL (optional)
###############

if (( $do_further_qhl == 1 )) 
then
    sh $analysis_script

    further_analyse_filename='analyse_further_qhl.sh'
    further_analysis_script="$this_run_directory$further_analyse_filename"
    let particles="$further_qhl_factor * $prt"
    let experiments="$further_qhl_factor * $exp"
    echo "------ Launching further QHL instance(s) ------"
    let max_qmd_id="$num_instances + 1"

    # write to a script so we can recall analysis later.
    cd $this_run_directory
    cd ../../../

    for i in \`seq 1 $max_qmd_id\`;
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
            -ggr=$growth_rule \
            $growth_rules_command 
    done
    echo "
    cd $this_run_directory
    python3 ../../../../scripts/AnalyseMultipleQMD.py \
        -dir=$this_run_directory \
        --bayes_csv=$bayes_csv \
        -log=$log_for_entire_run \
        -top=$number_best_models_further_qhl \
        -qhl=0 \
        -fqhl=1 \
        -true_expec=$system_measurements_file \
        -ggr=$growth_rule \
        -plot_probes=$plot_probe_file \
        -params=$run_info_file \
        -latex=$latex_mapping_file
    " > $further_analysis_script

    chmod a+x $further_analysis_script
    echo "------ Launching analyse further QHL ------"
    # sh $further_analysis_script
fi

# redis-cli shutdown
