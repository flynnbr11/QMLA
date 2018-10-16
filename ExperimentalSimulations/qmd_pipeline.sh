#!/bin/bash

test_description="qmd_runs"

### ---------------------------------------------------###
# Running QMD essentials
### ---------------------------------------------------###
num_tests=2
qhl_test=0
do_further_qhl=0

### ---------------------------------------------------###
# QHL parameters
### ---------------------------------------------------###
prt=20
exp=5
pgh=0.3
ra=0.8
rt=0.5
gaussian=1

### ---------------------------------------------------###
# QMD settings
### ---------------------------------------------------###
exp_data=1
sim_measurement_type='full_access'
exp_measurement_type='hahn' # to use if not experimental
measurement_type=$sim_measurement_type
# Overwrite settings for specific cases
"""
if (( "$exp_data" == 1))
then
    measurement_type=$exp_measurement_type
else
    measurement_type=$sim_measurement_type
fi
"""

if (( "$exp_data" == 0 ))
then
    pgh=0.3
fi

if (( "$qhl_test" == 1 ))
then
    pgh=1.0
fi

growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
# growth_rule='two_qubit_ising_rotation_hyperfine'
# growth_rule='test_multidimensional' 
use_rq=0
further_qhl_factor=2
plots=1
use_rq=0
number_best_models_further_qhl=5
custom_prior=1
dataset='NVB_dataset.p'
#dataset='NV05_dataset.p'
data_max_time=5000 # nanoseconds
data_time_offset=205 # nanoseconds

### ---------------------------------------------------###
# Everything from here downwards uses the parameters
# defined above to run QMD. 
### ---------------------------------------------------###
let max_qmd_id="$num_tests"
day_time=$(date +%b_%d/%H_%M)
directory="$day_time/"

cwd=$(pwd)
long_dir="$cwd/Results/$day_time/"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$long_dir$true_expec_filename"


this_log="$long_dir/qmd.log"
furhter_qhl_log="$long_dir/qhl_further.log"
mkdir -p $long_dir

true_operator='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
# true_operator='xTiPPyTiPPyTy'
# true_operator='xTyTTyTTTzTTTTiPPPPPxTyTTyTTTyTTTTyPPPPPxTyTTzTTTxTTTTiPPPPPiTyTTiTTTyTTTTy'
# true_operator='x'

declare -a qhl_operators=(
    $true_operator
)
declare -a particle_counts=(
$prt
)
q_id=0
if (($prt > 50)) || (($exp > 10)) 
then
    use_rq=1
else
    use_rq=0
fi

if (( $qhl_test == 1 )) # For QHL test always do without rq
then
    use_rq=0
fi
let bt="$exp-1"

printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log
# Launch $num_tests instances of QMD 
prior_pickle_file="$long_dir/prior.p"
true_params_pickle_file="$long_dir/true_params.p"

python3 ../Libraries/QML_lib/SetQHLParams.py \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -op=$true_operator \
    -exp=$exp_data \
    -rand_t=1 -rand_p=0 # can make true params and prior random


for prt in  "${particle_counts[@]}";
do
    for i in `seq 1 $max_qmd_id`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py \
            -op=$true_operator -p=$prt -e=$exp -bt=$bt \
            -rq=$use_rq -g=$gaussian -qhl=$qhl_test \
            -ra=$ra -rt=$rt -pgh=$pgh \
            -dir=$long_dir -qid=$q_id -pt=$plots -pkl=1 \
            -log=$this_log -cb=$bayes_csv \
            -meas=$measurement_type \
            -exp=$exp_data -cpr=$custom_prior \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -ds=$dataset -dst=$data_max_time -dto=$data_time_offset \
            -ggr=$growth_rule
    done
done



##
# Analyse results of QMD. (Only after QMD, not QHL).
##

analyse_filename='analyse.sh'
analyse_script="$long_dir$analyse_filename"

# write to a script so we can recall analysis later.
echo "
cd $long_dir
python3 ../../../../Libraries/QML_lib/AnalyseMultipleQMD.py \
    -dir=$long_dir --bayes_csv=$bayes_csv \
    -top=$number_best_models_further_qhl \
    -qhl=$qhl_test -fqhl=0 \
    -exp=$exp_data -true_expec=$true_expec_path \
    -ggr=$growth_rule \
    -data=$dataset -params=$true_params_pickle_file 
" > $analyse_script

chmod a+x $analyse_script
sh $analyse_script

if (( $do_further_qhl == 1 )) 
then
    for i in `seq 1 $max_qmd_id`;
        do
        pgh=0.3 # train on different set of data
        redis-cli flushall 
        let q_id="$q_id+1"
        let particles="$further_qhl_factor * $prt"
        let experiments="$further_qhl_factor * $exp"
    #    let q_id="$q_id+1"
        python3 Exp.py \
            -fq=$do_further_qhl \
            -p=$particles -e=$experiments -bt=$bt \
            -rq=$use_rq -g=$gaussian -qhl=0 \
            -ra=$ra -rt=$rt -pgh=1.0 \
            -dir=$long_dir -qid=$q_id -pt=$plots -pkl=1 \
            -log=$this_log -cb=$bayes_csv \
            -meas=$measurement_type \
            -exp=$exp_data -cpr=$custom_prior \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -ds=$dataset -dst=$data_max_time -dto=$data_time_offset \
            -ggr=$growth_rule
    done

    further_analyse_filename='analyse.sh'
    further_analyse_script="$long_dir$further_analyse_filename"

    # write to a script so we can recall analysis later.
    echo "
    cd $long_dir
    python3 ../../../../Libraries/QML_lib/AnalyseMultipleQMD.py \
        -dir=$long_dir --bayes_csv=$bayes_csv \
        -top=$number_best_models_further_qhl \
        -qhl=$qhl_test -fqhl=1 \
        -ggr=$growth_rule \
        -exp=$exp_data -true_expec=$true_expec_path \
        -data=$dataset -params=$true_params_pickle_file 
    " > $further_analyse_script

    chmod a+x $further_analyse_script
    sh $further_analyse_script
fi

