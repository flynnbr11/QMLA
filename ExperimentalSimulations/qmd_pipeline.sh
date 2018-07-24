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
qhl_test=1
q_id=0


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
            python3 Exp.py -p=7 -e=3 -rq=0 -ra=0.99 -g=0 -qhl=$qhl_test -op="$single_qubit" -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=1
        done 
    done

else
    for i in `seq 1 1`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -p=5 -e=3 -rq=0 -g=1 -qhl=$qhl_test -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=0
    done

    cd ../Libraries/QML_lib
    python3 AnalyseMultipleQMD.py -dir=$long_dir --bayes_csv=$bayes_csv

fi 


# TODO google array job for PBS -- node exclusive flag
