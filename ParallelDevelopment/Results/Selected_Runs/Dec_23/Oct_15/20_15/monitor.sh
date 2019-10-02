 

#!/bin/bash

echo "inside monitor script."
IFS=$'\n' read -d '' -r -a job_ids_s < /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/job_ids_started.txt
num_jobs_started=${#job_ids_s[@]}

IFS=$'\n' read -d '' -r -a job_ids_c < /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/job_ids_completed.txt
num_jobs_complete=${#job_ids_c[@]}

for k in ${job_ids_c[@]}
do
	echo $k
done


echo "num jobs started/finished: $num_jobs_started $num_jobs_complete"

while (( $num_jobs_complete < 100 ))
do
	IFS=$'\n' read -d '' -r -a job_ids_s < /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/job_ids_started.txt
	num_jobs_started=${#job_ids_s[@]}

	IFS=$'\n' read -d '' -r -a job_ids_c < /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/job_ids_completed.txt
	num_jobs_complete=${#job_ids_c[@]}

	echo "Waiting. Currently $num_jobs_complete / $num_jobs_started "
	sleep 3
done

sh /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/FINALISE_optimised_params_long_qmd_exp_data.sh

