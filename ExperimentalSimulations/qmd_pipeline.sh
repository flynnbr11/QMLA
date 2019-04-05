#!/bin/bash

test_description="multiple_growth_rules_include_hubbard"
printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log

### ---------------------------------------------------###
# Running QMD essentials
### ---------------------------------------------------###
num_tests=1
qhl_test=0
multiple_qhl=0
do_further_qhl=0
exp_data=0
simulate_experiment=0
q_id=0 # can start from other ID if desired

### ---------------------------------------------------###
# QHL parameters
### ---------------------------------------------------###
prt=200
exp=20
pgh=1.0
pgh_exponent=1.0
ra=0.8
rt=0.5

### ---------------------------------------------------###
# QMD settings
### ---------------------------------------------------###
use_rq=0
further_qhl_factor=1
further_qhl_num_runs=$num_tests
plots=1
number_best_models_further_qhl=5
custom_prior=1
bintimes=1
bf_all_times=0
data_max_time=5000 # nanoseconds
data_time_offset=205 # nanoseconds

### ---------------------------------------------------###
# Everything from here downwards uses the parameters
# defined above to run QMD. 
### ---------------------------------------------------###
let max_qmd_id="$num_tests + $q_id"

# Files where output will be stored
cwd=$(pwd)
day_time=$(date +%b_%d/%H_%M)
long_dir="$cwd/Results/$day_time/"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"
true_expec_filename="true_expec_vals.p"
true_expec_path="$long_dir$true_expec_filename"
prior_pickle_file="$long_dir/prior.p"
true_params_pickle_file="$long_dir/true_params.p"
plot_probe_file="$long_dir/plot_probes.p"
latex_mapping_filename='LatexMapping.txt'
latex_mapping_file=$long_dir$latex_mapping_filename
analyse_filename='analyse.sh'
analyse_script="$long_dir$analyse_filename"
this_log="$long_dir/qmd.log"
further_qhl_log="$long_dir/qhl_further.log"
mkdir -p $long_dir

# Choose a growth rule This will determine how QMD proceeds. 
# use_alt_growth_rules=1 # note this is redundant locally, currently

# sim_growth_rule='test_changes_to_qmd'
# sim_growth_rule='two_qubit_ising_rotation_hyperfine'
# sim_growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
# sim_growth_rule='non_interacting_ising'
# sim_growth_rule='non_interacting_ising_single_axis'
# sim_growth_rule='deterministic_noninteracting_ising_single_axis'
# sim_growth_rule='interacing_nn_ising_fixed_axis' 
# sim_growth_rule='interacting_nearest_neighbour_ising'
# sim_growth_rule='deterministic_interacting_nn_ising_single_axis'
# sim_growth_rule='deterministic_transverse_ising_nn_fixed_axis'
# sim_growth_rule='ising_1d_chain'
# sim_growth_rule='heisenberg_nontransverse'
# sim_growth_rule='heisenberg_transverse'
# sim_growth_rule='heisenberg_xyz'
# sim_growth_rule='hubbard'
# sim_growth_rule='hubbard_chain_just_hopping'
# sim_growth_rule='hubbard_chain'
#sim_growth_rule='hubbard_square_lattice_generalised'
sim_growth_rule='hopping_topology'

### Experimental growth rules 
### which will overwrite growth_rule if exp_data==1

# exp_growth_rule='two_qubit_ising_rotation_hyperfine'
# exp_growth_rule='two_qubit_ising_rotation_hyperfine_transverse'
exp_growth_rule='NV_centre_spin_large_bath'
# exp_growth_rule='NV_centre_experiment_debug'
# exp_growth_rule='reduced_nv_experiment'
# exp_growth_rule='PT_Effective_Hamiltonian'

if (( $exp_data == 1 )) || (( $simulate_experiment == 1 ))
then
    growth_rule=$exp_growth_rule
else
    growth_rule=$sim_growth_rule
fi
echo "SETTING GROWTH RULE TO: $growth_rule"

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

num_probes=10
force_plot_plus=0
gaussian=1
probe_noise=0.0000001
param_min=0
param_max=1
param_mean=0.5
param_sigma=3
rand_true_params=0
reallocate_resources=1
updater_from_prior=0
store_prt_wt=0 # store all particles and weights after learning

# rand_prior:
# if set to False (0), then uses any params specically 
# set in SetQHLParams dictionaries.
# All undefined params will be random according 
# to above defined mean/sigmas
rand_prior=1
special_probe='random' #'plus' #'ideal'
special_probe_plot='random'
# special_probe='plus_random' #'plus' #'ideal'

if (( "$exp_data" == 1)) || (( "$simulate_experiment" == 1)) 
then
#    special_probe='plus'
    # param_min=10
    # param_max=20
    # rand_true_params=0
    # special_probe='plus_random'
    # special_probe='plus'
    # special_probe='dec_13_exp'
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

echo "special probe $special_probe"
echo "growth rule: $growth_rule"



declare -a particle_counts=(
$prt
)

if (( $qhl_test == 1 )) # For QHL test always do without rq
then
    use_rq=0
fi
let bt="$exp"


# Launch $num_tests instances of QMD 

# First set up parameters/data to be used by all instances of QMD for this run. 
python3 ../Libraries/QML_lib/SetQHLParams.py \
    -true=$true_params_pickle_file \
    -prior=$prior_pickle_file \
    -probe=$plot_probe_file \
    -plus=$force_plot_plus \
    -pnoise=$probe_noise \
    -sp=$special_probe_plot \
    -ggr=$growth_rule \
    -exp=$exp_data \
    -g=$gaussian \
    -min=$param_min \
    -max=$param_max \
    -mean=$param_mean \
    -sigma=$param_sigma \
    -rand_t=$rand_true_params \
    -rand_p=$rand_prior \


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
            -p=$prt -e=$exp -bt=$bt \
            -rq=$use_rq -g=$gaussian -qhl=$qhl_test \
            -ra=$ra -rt=$rt -pgh=$pgh \
            -pgh_exp=$pgh_exponent \
            -dir=$long_dir -qid=$q_id -pt=$plots -pkl=1 \
            -log=$this_log -cb=$bayes_csv \
            -exp=$exp_data -cpr=$custom_prior \
            -prtwt=$store_prt_wt \
            -nprobes=$num_probes \
            -pnoise=$probe_noise \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -special_probe=$special_probe \
            -pmin=$param_min -pmax=$param_max \
            -pmean=$param_mean -psigma=$param_sigma \
            -dst=$data_max_time \
            -bintimes=$bintimes \
            -bftimesall=$bf_all_times \
            -dto=$data_time_offset \
            -latex=$latex_mapping_file \
            -resource=$reallocate_resources \
            --updater_from_prior=$updater_from_prior \
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
            -exp=$exp_data -cpr=$custom_prior \
            -prior_path=$prior_pickle_file \
            -true_params_path=$true_params_pickle_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -dst=$data_max_time \
            -dto=$data_time_offset \
            -latex=$latex_mapping_file \
            -ggr=$growth_rule \
            --updater_from_prior=$updater_from_prior

    done

    cd $long_dir
    python3 ../../../../Libraries/QML_lib/AnalyseMultipleQMD.py \
        -dir=$long_dir --bayes_csv=$bayes_csv \
        -top=$number_best_models_further_qhl \
        -qhl=$qhl_test \
        -fqhl=1 \
        -ggr=$growth_rule \
        -exp=$exp_data \
        -true_expec=$true_expec_path \
        -plot_probes=$plot_probe_file \
        -params=$true_params_pickle_file \
        -latex=$latex_mapping_file
    " > $further_analyse_script

    chmod a+x $further_analyse_script
    sh $further_analyse_script
fi

