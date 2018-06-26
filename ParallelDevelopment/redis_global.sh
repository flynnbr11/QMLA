#!/bin/bash

./just_launch_redis.sh

echo "testing redis server in redis global script"
module load tools/redis-4.0.8
redis-cli ping

global_host="$(hostname)"
echo " In redis global script, global_host=$global_host"

qsub -v GLOBAL_SERVER=$global_host global_host_remote.sh

#for i in `seq 1 1`;
#do
#	qsub -v GLOBAL_SERVER=$global_host global_host_remote.sh
#done 


