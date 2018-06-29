#!/bin/bash

max_qmd_id=10
directory="multtestdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"

this_log="$cwd/graph_dev.log"

rm -r $long_dir
rm $this_log
mkdir -p $long_dir

q_id=0
for i in `seq 1 2`;
do
    for j in `seq 1 1`;
    do
        let num_prt="$i+10"
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -p 150 -e 3 -rq=0 -dir=$directory -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv 
    done 
done

cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=$long_dir --bayes_csv=$bayes_csv


# TODO google array job for PBS -- node exclusive flag
