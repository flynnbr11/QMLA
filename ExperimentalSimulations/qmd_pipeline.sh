#!/bin/bash

max_qmd_id=10
directory="multtestdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
rm -r $long_dir
mkdir -p $long_dir

for i in `seq 1 $max_qmd_id`;
do
    let num_prt="$i+10"
    redis-cli flushall
    python3 Exp.py -p $num_prt -e 4 -rq=0 -dir=$directory -qid=$i
done 


cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=$long_dir


# TODO google array job for PBS -- node exclusive flag
