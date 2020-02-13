#!/bin/bash

test_description="development"
printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log

### ---------------------------------------------------###
# Running QMD essentials
### ---------------------------------------------------###
num_tests=1
qhl_test=0 # don't perform QMLA; perform QHL on known correct model
multiple_qhl=0 # perform QHL for defined list of models.
do_further_qhl=0 # QHL refinement to best performing models 
exp_data=0
simulate_experiment=0
q_id=0 # can start from other ID if desired

### ---------------------------------------------------###
# QHL parameters
### --------------------------------------------------###
prt=20
exp=2
pgh=1.0
pgh_exponent=1.0
pgh_increase=0 # whether to add to time found by PGH (bool)
ra=0.98
rt=0.5

### ---------------------------------------------------###
# QMD settings
### ---------------------------------------------------###
use_rq=0
further_qhl_factor=1
further_qhl_num_runs=$num_tests
plots=0
number_best_models_further_qhl=5
custom_prior=1
bintimes=1
bf_all_times=0
# data_max_time=5 # nanoseconds
# data_time_offset=205 # nanoseconds

### ---------------------------------------------------###
# Everything from here downwards uses the parameters
# defined above to run QMD. 
### ---------------------------------------------------###
let max_qmd_id="$num_tests + $q_id"

# Files where output will be stored
cwd=$(pwd)
day_time=$(date +%b_%d/%H_%M)
full_path_to_results="$cwd/Results/$day_time/"
running_dir="$(pwd)"
qmd_dir="${running_dir%/ExperimentalSimulations}" # chop off ParallelDevelopment to get qmd folder path
lib_dir="$qmd_dir/Libraries/QML_lib"
bcsv="cumulative.csv"
bayes_csv="$full_path_to_results$bcsv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$full_path_to_results$true_expec_filename"
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

# Choose a growth rule This will determine how QMD proceeds. 
# use_alt_growth_rules=1 # note this is redundant locally, currently

# sim_growth_rule='IsingProbabilistic'
# sim_growth_rule='IsingPredetermined'
# sim_growth_rule='HeisenbergXYZPredetermined'
sim_growth_rule='HeisenbergXYZProbabilistic'
# sim_growth_rule='FermiHubbardPredetermined'
# sim_growth_rule='FermiHubbardProbabilistic'
# sim_growth_rule='Genetic'
# sim_growth_rule='Presentation'
# sim_growth_rule='ExperimentReducedNV'

### Experimental growth rules 
### which will overwrite growth_rule if exp_data==1

exp_growth_rule='ExperimentNVCentre'
# exp_growth_rule='ExperimentNVCentreNoTransvereTerms'
# exp_growth_rule='ExpAlternativeNV'
# exp_growth_rule='ExperimentFullAccessNV'
# exp_growth_rule='NVLargeSpinBath'
# exp_growth_rule='ExperimentNVCentreVaryTrueModel'
# exp_growth_rule='ExpNVRevivals'
# exp_growth_rule='ExperimentReducedNV'


if (( $exp_data == 1 )) || (( $simulate_experiment == 1 ))
then
    growth_rule=$exp_growth_rule
else
    growth_rule=$sim_growth_rule
fi

alt_growth_rules=(
    # 'IsingPredetermined'
    # 'IsingProbabilistic'
    # 'HeisenbergXYZProbabilistic'
    # 'HeisenbergXYZPredetermined'
    # 'FermiHubbardPredetermined' 
    # 'FermiHubbardProbabilistic' 
    # 'ExperimentReducedNV'
    # 'ExperimentNVCentre'
)

growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done

num_probes=10
force_plot_plus=0
gaussian=1
probe_noise=0.0000001
param_min=0
param_max=10
param_mean=0.5
param_sigma=3
rand_true_params=0
reallocate_resources=0
updater_from_prior=0
store_prt_wt=0 # store all particles and weights after learning

# rand_prior:
# if set to False (0), then uses any params specically 
# set in SetQHLParams dictionaries.
# All undefined params will be random according 
# to above defined mean/sigmas
rand_prior=0
special_probe='random' #'plus' #'ideal'
special_probe_plot='plus' #'random'

if (( "$exp_data" == 1))  
then
    special_probe='dec_13_exp'
    special_probe_plot='plus'
elif (( "$simulate_experiment" == 1)) 
then
    special_probe='random'
    special_probe_plot='plus'
fi




if [[ "$growth_rule" == "PT_Effective_Hamiltonian" ]] 
then
    echo "In if statement for PT_Effective_Hamiltonian"
    special_probe='None'
    special_probe_plot='None'
fi

# measurement_type=$exp_measurement_type
# special_probe='plus' #'plus' #'ideal' # TODO this is just for a test, remove!!

declare -a particle_counts=(
$prt
)

let bt="$exp"


# Launch $num_tests instances of QMD 

# First set up parameters/data to be used by all instances of QMD for this run. 
# python3 ../qmla/SetQHLParams.py \
python3 ../Scripts/set_qmla_params.py \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -probe=$plot_probe_file \
    -pnoise=$probe_noise \
    -ggr=$growth_rule \
    -exp=$exp_data \
    -g=$gaussian \
    -mean=$param_mean \
    -sigma=$param_sigma \
    -rand_t=$rand_true_params \
    -rand_p=$rand_prior \
    -dir=$full_path_to_results \
    -log=$this_log \
    -min=$param_min \
    -max=$param_max \
    -plus=$force_plot_plus \
    -sp=$special_probe_plot \
    $growth_rules_command 

echo "Generated configuration."

for prt in  "${particle_counts[@]}";
do
    for i in `seq 1 $max_qmd_id`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        python3 \
            ../Scripts/implement_qmla.py \
            -qhl=$qhl_test \
            -mqhl=$multiple_qhl \
            -rq=$use_rq \
            -p=$prt \
            -e=$exp \
            -bt=$bt \
            -ra=$ra \
            -rt=$rt \
            -qid=$q_id \
            -log=$this_log \
            -dir=$full_path_to_results \
            -pgh=$pgh \
            -pgh_exp=$pgh_exponent \
            -pgh_incr=$pgh_increase \
            -pt=$plots \
            -pkl=1 \
            -cb=$bayes_csv \
            -exp=$exp_data \
            -prtwt=$store_prt_wt \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -latex=$latex_mapping_file \
            -resource=$reallocate_resources \
            -ggr=$growth_rule \
            $growth_rules_command 
    done
done

echo "

------ QMLA completed ------

"

##
# Analyse results of QMD. (Only after QMD, not QHL).
##


# write to a script so we can recall analysis later.
echo "
cd $full_path_to_results
python3 ../../../../Scripts/analyse_qmla.py \
    -dir=$full_path_to_results --bayes_csv=$bayes_csv \
    -log=$this_log \
    -top=$number_best_models_further_qhl \
    -qhl=$qhl_test -fqhl=0 \
    -exp=$exp_data -true_expec=$true_expec_path \
    -ggr=$growth_rule \
    -plot_probes=$plot_probe_file \
    -params=$true_params_pickle_file \
    -latex=$latex_mapping_file

python3 ../../../../Scripts/generate_results_pdf.py \
    -dir=$full_path_to_results \
    -p=$prt -e=$exp -bt=$bt -t=$num_tests \
    -log=$this_log \
    -nprobes=$num_probes \
    -pnoise=$probe_noise \
    -special_probe=$special_probe \
    -ggr=$growth_rule \
    -run_desc=$test_description \
    -git_commit=$git_commit \
    -ra=$ra \
    -rt=$rt \
    -pgh=$pgh \
    -qhl=$qhl_test \
    -mqhl=$multiple_qhl \
    -cb=$bayes_csv \
    -exp=$exp_data

" > $analyse_script


chmod a+x $analyse_script

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
        pgh=0.3 # train on different set of data
        redis-cli flushall 
        # let q_id=\"\$q_id+1\"
        # q_id=\$((q_id+1))
        let q_id="$q_id + 1"
        echo "QID: $q_id"
        python3 /Scripts/implement_qmla.py \
            -fq=1 \
            -p=$particles \
            -e=$experiments \
            -bt=$bt \
            -rq=$use_rq \
            -qhl=0 \
            -ra=$ra \
            -rt=$rt \
            -pgh=1.0 \
            -pgh_exp=$pgh_exponent \
            -pgh_incr=$pgh_increase \
            -dir=$full_path_to_results \
            -qid=$q_id \
            -pt=$plots \
            -pkl=1 \
            -log=$this_log \
            -cb=$bayes_csv \
            -exp=$exp_data \
            -pnoise=$probe_noise \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -latex=$latex_mapping_file \
            -ggr=$growth_rule \
            -resource=$reallocate_resources \
            -ggr=$growth_rule \
            $growth_rules_command 
    done
    echo "
    cd $full_path_to_results
    python3 ../../../../Scripts/AnalyseMultipleQMD.py \
        -dir=$full_path_to_results \
        --bayes_csv=$bayes_csv \
        -log=$this_log \
        -top=$number_best_models_further_qhl \
        -qhl=0 \
        -fqhl=1 \
        -exp=$exp_data \
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
