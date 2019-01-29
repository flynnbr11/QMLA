#!/bin/bash

test_description="multiple_growth_rules_include_hubbard"

### ---------------------------------------------------###
# Running QMD essentials
### ---------------------------------------------------###
num_tests=5
qhl_test=0
multiple_qhl=1
do_further_qhl=0
exp_data=0
simulate_experiment=0

### ---------------------------------------------------###
# QHL parameters
### ---------------------------------------------------###
prt=5
exp=3
pgh=1.0
ra=0.8
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
bintimes=0
# dataset='NVB_dataset.p'
dataset='NVB_rescale_dataset.p'

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
further_qhl_log="$long_dir/qhl_further.log"
mkdir -p $long_dir

# growth_rule='test_changes_to_qmd'
# use_alt_growth_rules=1 # note this is redundant locally, currently
# growth_rule='two_qubit_ising_rotation_hyperfine'
# growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
# growth_rule='non_interacting_ising'
# growth_rule='non_interacting_ising_single_axis'
# growth_rule='deterministic_noninteracting_ising_single_axis'
# growth_rule='interacing_nn_ising_fixed_axis' 
# growth_rule='interacting_nearest_neighbour_ising'
# growth_rule='deterministic_interacting_nn_ising_single_axis'
# growth_rule='deterministic_transverse_ising_nn_fixed_axis'
# growth_rule='ising_1d_chain'

# growth_rule='heisenberg_nontransverse'
# growth_rule='heisenberg_transverse'
growth_rule='heisenberg_xyz'

# growth_rule='hubbard'
# growth_rule='hubbard_chain_just_hopping'
# growth_rule='hubbard_chain'
#growth_rule='hubbard_square_lattice_generalised'


alt_growth_rules=(
   # 'heisenberg_transverse'
   # 'interacing_nn_ising_fixed_axis'
   # 'non_interacting_ising'
#  'ising_1d_chain'
 # 'hubbard_square_lattice_generalised'
)

growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done


# true_operator='yTi'
# true_operator='xTxTTiPPPiTxTTx'
# true_operator='xTxTTiTTTiPPPPiTxTTxTTTiPPPPiTiTTxTTTx'
# true_operator='xTxTTiTTTiTTTTiPPPPPiTxTTxTTTiTTTTiPPPPPiTiTTxTTTxTTTTi'
# true_operator='xTxTTiTTTiTTTTiPPPPPiTxTTxTTTiTTTTiPPPPPiTiTTxTTTxTTTTiPPPPPiTiTTiTTTxTTTTx'
true_operator='xTxTTx'
# true_operator='xTiPPyTiPPyTy'
# true_operator='xTyTTyTTTzTTTTiPPPPPxTyTTyTTTyTTTTyPPPPPxTyTTzTTTxTTTTiPPPPPiTyTTiTTTyTTTTy'

qhl_operators=(
    'xTx'
    'yTy'
)

sim_measurement_type='full_access' # measurement to use during simulated cases. 
exp_measurement_type='hahn' # to use if not experimental
if (( "$exp_data" == 1))
then
    measurement_type=$exp_measurement_type
    # pgh=0.3
    true_operator='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
    growth_rule='two_qubit_ising_rotation_hyperfine'
    # growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
elif (( "$simulate_experiment" == 1)) 
then
    measurement_type=$exp_measurement_type
    # pgh=0.3
    true_operator='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
    growth_rule='two_qubit_ising_rotation_hyperfine'
else
    measurement_type=$sim_measurement_type
fi

declare -a qhl_operators=(
    $true_operator
)
declare -a particle_counts=(
$prt
)
q_id=0
"""
if (($prt > 50)) || (($exp > 10)) 
then
    use_rq=1
else
    use_rq=0
fi
"""
if (( $qhl_test == 1 )) # For QHL test always do without rq
then
    use_rq=0
fi
# use_rq=1
# let bt="$exp-1"
let bt="$exp"

printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log
# Launch $num_tests instances of QMD 
prior_pickle_file="$long_dir/prior.p"
true_params_pickle_file="$long_dir/true_params.p"
plot_probe_file="$long_dir/plot_probes.p"
force_plot_plus=0
gaussian=1
param_min=0
param_max=8
param_mean=0.5
param_sigma=2
rand_true_params=0
# rand_prior:
# if set to False (0), then uses any params specically 
# set in SetQHLParams dictionaries.
# All undefined params will be random according 
# to above defined mean/sigmas
rand_prior=0 
special_probe='random' #'plus' #'ideal'
# special_probe='plus_random' #'plus' #'ideal'
if (( "$exp_data" == 1))
then
#    special_probe='plus'
    special_probe='plus_random'
elif (( "$simulate_experiment" == 1)) 
then 
    special_probe='plus_random'
fi

# measurement_type=$exp_measurement_type
# special_probe='plus' #'plus' #'ideal' # TODO this is just for a test, remove!!

echo "special probe $special_probe"

python3 ../Libraries/QML_lib/SetQHLParams.py \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -probe=$plot_probe_file \
    -plus=$force_plot_plus \
    -sp=$special_probe \
    -ggr=$growth_rule \
    -op=$true_operator \
    -exp=$exp_data \
    -g=$gaussian \
    -min=$param_min \
    -max=$param_max \
    -mean=$param_mean \
    -sigma=$param_sigma \
    -rand_t=$rand_true_params \
    -rand_p=$rand_prior # can make true params and prior random


latex_mapping_filename='LatexMapping.txt'
latex_mapping_file=$long_dir$latex_mapping_filename
reallocate_resources=0

for prt in  "${particle_counts[@]}";
do
    for i in `seq 1 $max_qmd_id`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        # python3 -m cProfile \
            # -o "Profile_linalg_long_run.txt" \
        python3 \
            Exp.py \
            -mqhl=$multiple_qhl \
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
            -plot_probes=$plot_probe_file \
            -special_probe=$special_probe \
            -pmin=$param_min -pmax=$param_max \
            -pmean=$param_mean -psigma=$param_sigma \
            -ds=$dataset -dst=$data_max_time \
            -bintimes=$bintimes \
            -dto=$data_time_offset \
            -latex=$latex_mapping_file \
            -resource=$reallocate_resources \
            -ggr=$growth_rule \
            $growth_rules_command 
    done
done

echo "

------ QMD finished learning ------

"

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
    -plot_probes=$plot_probe_file \
    -data=$dataset \
    -params=$true_params_pickle_file \
    -latex=$latex_mapping_file
" > $analyse_script

chmod a+x $analyse_script
# sh $analyse_script

if (( $do_further_qhl == 1 )) 
then
    further_analyse_filename='analyse_further_qhl.sh'
    further_analyse_script="$long_dir$further_analyse_filename"
    let particles="$further_qhl_factor * $prt"
    let experiments="$further_qhl_factor * $exp"

    # write to a script so we can recall analysis later.
    echo "
    cd $long_dir
    cd ../../../

    for i in \`seq 1 $max_qmd_id\`;
        do
        pgh=0.3 # train on different set of data
        redis-cli flushall 
        #let q_id=\"\$q_id+1\"
        q_id=\$((q_id+1))
        python3 Exp.py \
            -fq=$do_further_qhl \
            -p=$particles -e=$experiments -bt=$bt \
            -rq=$use_rq -g=$gaussian -qhl=0 \
            -ra=$ra -rt=$rt -pgh=1.0 \
            -dir=$long_dir -qid=\$q_id -pt=$plots -pkl=1 \
            -log=$this_log -cb=$bayes_csv \
            -meas=$measurement_type \
            -exp=$exp_data -cpr=$custom_prior \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -ds=$dataset -dst=$data_max_time \
            -dto=$data_time_offset \
            -latex=$latex_mapping_file \
            -ggr=$growth_rule \

    done

    cd $long_dir
    python3 ../../../../Libraries/QML_lib/AnalyseMultipleQMD.py \
        -dir=$long_dir --bayes_csv=$bayes_csv \
        -top=$number_best_models_further_qhl \
        -qhl=$qhl_test \
        -fqhl=1 \
        -ggr=$growth_rule \
        -meas=$measurement_type \
        -exp=$exp_data \
        -true_expec=$true_expec_path \
        -plot_probes=$plot_probe_file \
        -data=$dataset \
        -params=$true_params_pickle_file \
        -latex=$latex_mapping_file
    " > $further_analyse_script

    chmod a+x $further_analyse_script
    sh $further_analyse_script
fi

