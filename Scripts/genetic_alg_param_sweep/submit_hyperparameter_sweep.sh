#!/bin/bash

time="walltime=00:15:00"
processes_request="nodes=1:ppn=4"

qsub -l $processes_request,$time run_my_script.sh

