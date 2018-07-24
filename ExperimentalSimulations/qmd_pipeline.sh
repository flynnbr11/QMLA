#!/bin/bash


test_description="QHL, non-Gaussian 5000prt;1500exp"

max_qmd_id=10
day_time=$(date +%b_%d/%H_%M)
directory="$day_time/"

cwd=$(pwd)
long_dir="$cwd/Results/$day_time/"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"

this_log="$long_dir/qmd.log"

# rm -r $long_dir
# rm $this_log
mkdir -p $long_dir
#mkdir -p $directory

"""
declare -a qhl_operators=(
'xTi'
'yTi'
'zTi'
'xTiPPyTi'
'xTiPPzTi'
'xTiPPzTiPPyTi'
'xTiPPzTiPPyTiPPxTx'
'xTiPPzTiPPyTiPPyTy'
'xTiPPzTiPPyTiPPzTz'
'xTiPPzTiPPyTiPPzTzPPxTx'
'xTiPPzTiPPyTiPPzTzPPyTy'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTy'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPxTz'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPxTy'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPyTz'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPxTyPPxTz'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPxTyPPyTz'
'xTiPPzTiPPyTiPPzTzPPxTxPPyTyPPxTyPPyTzPPxTz' 
)

"""
declare -a qhl_operators=(
'z'
)

true_operator='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
two_param='zTiPPxTi'
one_param='zTi'
single_qubit='x'
sample='xTiPPzTiPPyTy'
qhl_test=0
q_id=0
exp_data=1
prt=1000
exp=400
gaussian=1


printf "$day_time: \t $test_description \n" >> QMD_Results_directories.log

if [ "$qhl_test" == 1 ]
then
    for op in "${qhl_operators[@]}";
    do
        for i in `seq 1 1`;
        do
            let num_prt="$i+10"
            redis-cli flushall
            let q_id="$q_id+1"
            python3 Exp.py -p=$prt -e=$exp -rq=0 -ra=0.99 -g=$gaussian -qhl=$qhl_test -op="$one_param" -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=$exp_data
        done 
    done

else
    for i in `seq 1 1`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -p=$prt -e=$exp -rq=0 -g=$gaussian -qhl=$qhl_test -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=$exp_data
    done

    cd ../Libraries/QML_lib
    python3 AnalyseMultipleQMD.py -dir=$long_dir --bayes_csv=$bayes_csv

fi 


# TODO google array job for PBS -- node exclusive flag
