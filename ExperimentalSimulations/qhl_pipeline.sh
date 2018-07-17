#!/bin/bash

max_qmd_id=10
directory="multtestdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
bcsv="cumulative.csv"
bayes_csv="$long_dir$bcsv"

this_log="$cwd/dev_exp.log"

rm -r $long_dir
rm $this_log
mkdir -p $long_dir
mkdir -p $directory

q_id=0
for i in `seq 1 1`;
do
    for j in `seq 1 1`;
    do
        let num_prt="$i+10"
        redis-cli flushall
        let q_id="$q_id+1"
        python3 test_qhl.py -p=15 -e=5 -op='xTi' -rq=0 -g=0 -dir=$long_dir -qid=$q_id -pt=1 -pkl=1 -log=$this_log -cb=$bayes_csv -exp=0
    done 
done

