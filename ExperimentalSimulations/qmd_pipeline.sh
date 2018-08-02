#!/bin/bash

test_description="QHL, non-Gaussian 5000prt;1500exp"
num_tests=1
let max_qmd_id="$num_tests"

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
exp_data=0
prt=150
exp=20
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
            python3 Exp.py -p=$prt -e=$exp -rq=0 -ra=0.99 -g=$gaussian -qhl=$qhl_test -op="$true_operator" -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=$exp_data
        done 
    done

else
    for i in `seq 1 $max_qmd_id`;
    do
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -op='xTi' -p=$prt -e=$exp -rq=0 -g=$gaussian -qhl=$qhl_test -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=$exp_data
    done
    cd ../Libraries/QML_lib
    
    if [ $num_tests > 1 ]
    then
        python3 AnalyseMultipleQMD.py -dir=$long_dir --bayes_csv=$bayes_csv
    fi

fi 


# TODO google array job for PBS -- node exclusive flag
