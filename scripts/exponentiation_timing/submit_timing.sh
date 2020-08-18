#!/bin/bash

time="walltime=02:30:00"
processes_request="nodes=1:ppn=1"

results_dir="$(pwd)/"
mkdir -p $results_dir

qsub -l $processes_request,$time -o "$results_dir/output.txt" -e "$results_dir/error.txt" run_timing.sh

