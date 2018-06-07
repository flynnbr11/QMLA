#!/bin/bash

max_qmd_id=10
directory="multtestdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
rm -r $long_dir
mkdir -p $long_dir

q_id=0
for i in `seq 1 5`;
do
    for j in `seq 1 1`;
    do
        let num_prt="$i+10"
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -p 10 -e 4 -rq=0 -dir=$directory -qid=$q_id -pt=1 -pkl=1 -log='highTfailure.log' -cb='cumulative.csv'
    done 
done

cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=$long_dir


# TODO google array job for PBS -- node exclusive flag
