#!/bin/bash
#PBS -l nodes=2:ppn=16,walltime=00:30:00
#PBS -q veryshort

# module load tools/redis-4.0.8
# module load mvapich/gcc/64/1.2.0-qlc

set -x

# The redis server is started on the first node.
SERVER_HOST=$(head -1 "$PBS_NODEFILE")
#TODO create a redis config

qmd
cd Libraries/QML_lib
redis-server --protected-mode no &

# wait for redis to start up
sleep 5

MPI_ARGS="-np 2 -ppn 1"


mpirun -np 1 -ppn 4 -print-rank-map -prepend-rank ./rq_worker_launch $SERVER_HOST & 
sleep 20


export QMD_REDIS_HOST=$SERVER_HOST

qmd 
cd ValidateQLE/

python3 ExperimentalSpawningRule.py 

