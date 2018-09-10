#!/bin/bash

echo 'inside monitor script.'
IFS=$'\n' read -d '' -r -a job_ids_s < $(pwd)/job_ids_started.txt
num_jobs_started=${#job_ids_s[@]}


for k in ${job_ids_s[@]}
do
	echo $k
done

IFS=$'\n' read -d '' -r -a job_ids_c < $(pwd)/job_ids_completed.txt
num_jobs_complete=${#job_ids_c[@]}

echo "num jobs started/finished: $num_jobs_started $num_jobs_complete"

while (( $num_jobs_started != $num_jobs_complete )) && (( $num_job_started != 0  ))
do
	IFS=$'\n' read -d '' -r -a job_ids_s < $(pwd)/job_ids_started.txt
	num_jobs_started=${#job_ids_s[@]}


	IFS=$'\n' read -d '' -r -a job_ids_c < $(pwd)/job_ids_completed.txt
	num_jobs_complete=${#job_ids_c[@]}

	echo 'Waiting. Currently $num_jobs_complete / $num_jobs_started'
	sleep 3
done
echo 'outside while loop'
