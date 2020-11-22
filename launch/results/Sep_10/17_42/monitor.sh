 

#!/bin/bash

echo "inside monitor script."
IFS=$'\n' read -d '' -r -a job_ids_s < /home/bf16951/QMD/Launch/Results/Sep_10/17_42//job_ids_started.txt
num_jobs_started=${#job_ids_s[@]}

IFS=$'\n' read -d '' -r -a job_ids_c < /home/bf16951/QMD/Launch/Results/Sep_10/17_42//job_ids_completed.txt
num_jobs_complete=${#job_ids_c[@]}

for k in ${job_ids_c[@]}
do
	echo $k
done


echo "num jobs started/finished: $num_jobs_started $num_jobs_complete"

while (( $num_jobs_complete < 5 ))
do
	IFS=$'\n' read -d '' -r -a job_ids_s < /home/bf16951/QMD/Launch/Results/Sep_10/17_42//job_ids_started.txt
	num_jobs_started=${#job_ids_s[@]}

	IFS=$'\n' read -d '' -r -a job_ids_c < /home/bf16951/QMD/Launch/Results/Sep_10/17_42//job_ids_completed.txt
	num_jobs_complete=${#job_ids_c[@]}

	echo "Waiting. Currently $num_jobs_complete / $num_jobs_started "
	sleep 3
done

sh /home/bf16951/QMD/Launch/Results/Sep_10/17_42//FINALISE_NV_bath-GR__2-qubit-taget__all-models-in-stage-at-once__qhl.sh

