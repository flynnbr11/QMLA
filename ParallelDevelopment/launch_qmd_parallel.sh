#!/bin/bash
#PBS -l nodes=1:ppn=4,walltime=00:30:00
#PBS -q testq

running_dir=$(pwd)
lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"

module load tools/redis-4.0.8
module load mvapich/gcc/64/1.2.0-qlc

set -x

# The redis server is started on the first node.
SERVER_HOST=$(head -1 "$PBS_NODEFILE")
# HOST='localhost'
REDIS_URL=redis://$SERVER_HOST:6379
echo "REDIS_URL is $REDIS_URL"
#TODO create a redis config

cd $lib_dir
redis-server --protected-mode no &

# mpirun -np 1 -ppn 4 rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &
rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &


cd $running_dir
python3 Exp.py -rq=1 -p=15 -e=5 -bt=2 -host=$HOST



sleep 1
echo "   SHUTDOWN REDIS   "
redis-cli shutdown

