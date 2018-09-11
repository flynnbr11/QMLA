#!/bin/bash

test_description="QHL, non-Gaussian 5000prt;1500exp"
num_tests=2
let max_qmd_id="$num_tests"

day_time=$(date +%b_%d/%H_%M)
directory="$day_time/"

cwd=$(pwd)
long_dir="$cwd/Results/$day_time/"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"

this_log="$long_dir/qmd.log"
furhter_qhl_log="$long_dir/qhl_further.log"

# rm -r $long_dir
# rm $this_log
mkdir -p $long_dir
#mkdir -p $directory


true_operator='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
declare -a qhl_operators=(
    $true_operator
)
qhl_test=1
do_further_qhl=0
top_number_models=3
q_id=0
exp_data=1
use_rq=0
prt=16
exp=10
further_qhl_factor=1
if (($prt > 50)) || (($exp > 10)) || (( $qhl_test == 0 ))
then
    use_rq=1
fi
use_rq=0
let bt="$exp-1"
pgh=0.3
ra=0.8
rt=0.5
gaussian=1
custom_prior=1
#dataset='NVB_HahnPeaks_Newdata'
dataset='NV05_HahnPeaks_expdataset'
data_max_time=3500 # nanoseconds
data_time_offset=205 # nanoseconds


printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log
# Launch $num_tests instances of QMD 

declare -a particle_counts=(
10
20
)

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
            -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 \
            -log=$this_log -cb=$bayes_csv \
            -exp=$exp_data -cpr=$custom_prior \
            -ds=$dataset -dst=$data_max_time -dto=$data_time_offset
    done
done 
# Analyse results of QMD. (Only after QMD, not QHL).
python3 ../Libraries/QML_lib/AnalyseMultipleQMD.py \
    -dir=$long_dir --bayes_csv=$bayes_csv \
    -top=$top_number_models -qhl=$qhl_test


if (( $do_further_qhl == 1 ))
then
    redis-cli flushall 
    let particles="$further_qhl_factor * $prt"
    let experiments="$further_qhl_factor * $exp"
#    let q_id="$q_id+1"
    python3 Exp.py \
        -fq=$do_further_qhl \
        -p=$particles -e=$experiments -bt=$bt \
        -rq=$use_rq -g=$gaussian -qhl=0 \
        -ra=$ra -rt=$rt -pgh=$pgh \
        -dir=$long_dir -qid=$q_id -pt=1 -pkl=0 \
        -log=$this_log -cb=$bayes_csv \
        -exp=$exp_data -cpr=$custom_prior \
        -ds=$dataset -dst=$data_max_time -dto=$data_time_offset
fi

