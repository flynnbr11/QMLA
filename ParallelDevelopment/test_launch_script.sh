#!/bin/bash


for i in `seq 1 5`;
do
	qsub launch_qmd_parallel.sh.cjw
done 
