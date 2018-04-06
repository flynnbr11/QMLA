#!/bin/bash
#PBS -l nodes=1:ppn=4,walltime=00:30:00
#PBS -q testq

module load tools/redis-4.0.8
module load mvapich/gcc/64/1.2.0-qlc

set -x

# The redis server is started on the first node.
SERVER_HOST=$(head -1 "$PBS_NODEFILE")
# HOST='127.0.0.1'
HOST='localhost'
REDIS_URL=redis://$HOST:6379
echo "REDIS_URL is $REDIS_URL"
#TODO create a redis config

cd Libraries/QML_lib
redis-server --protected-mode no &

# mpirun -np 1 -ppn 8 rq worker -u $REDIS_URL &
mpirun -np 1 -ppn 4 rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &


cd ../../ValidateQLE
python3 ExperimentalSpawningRule.py -rq=1 -p=15 -e=5 -bt=2 -host=$HOST



sleep 1
echo "   SHUTDOWN REDIS   "
redis-cli shutdown

