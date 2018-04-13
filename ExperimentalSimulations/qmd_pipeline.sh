#!/bin/bash

max_qmd_id=5
directory="testdir/"

cwd=$(pwd)
long_dir="$cwd/Results/$directory"
mkdir -p $long_dir


for i in `seq 1 $max_qmd_id`;
do
    python3 Exp.py -rq=0 -dir=$directory -qid=$i
done 


cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=$long_dir
