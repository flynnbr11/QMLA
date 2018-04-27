#!/bin/bash


for i in `seq 1 15`;
do
	qsub -v QMD_ID=$i launch_qmd_parallel.sh.cjw
done 
