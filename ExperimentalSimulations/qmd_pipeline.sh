#!/bin/bash

max_qmd_id=10
directory="multtestdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
rm -r $long_dir
mkdir -p $long_dir

q_id=0
for i in `seq 1 3`;
do
    for j in `seq 1 3`;
    do
        let num_prt="$i+10"
        redis-cli flushall
        let q_id="$q_id+1"
        python3 Exp.py -p $num_prt -e 4 -rq=0 -dir=$directory -qid=$q_id
    done 
done

cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=$long_dir


# TODO google array job for PBS -- node exclusive flag
