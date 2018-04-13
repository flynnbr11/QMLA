#!/bin/bash
#PBS -l nodes=2:ppn=3, walltime=06:00:00
#PBS -q veryshort

host=$(hostname)
echo "host= $host"

if [ "$host" == "IT067176" ]
then
    echo "host= $host"
    running_dir=$(pwd)
    lib_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/Libraries/QML_lib"
    script_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations"
    SERVER_HOST='localhost'
    ~/redis-4.0.8/src/redis-server & 
        
elif [ "$host" == "newblue4" ]
then
    echo "BC frontend identified"
    echo "host= $host"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load mvapich/gcc/64/1.2.0-qlc
    echo "launching redis"
    redis-server --protected-mode no &
    SERVER_HOST='localhost'


elif [[ "$host" == "node"* ]]
then
    echo "BC backend identified"
    echo "host= $host"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load languages/intel-compiler-16-u2
    SERVER_HOST=$(head -1 "$PBS_NODEFILE")
    echo "launching redis"
    redis-server --protected-mode no &


else
    echo "Neither local machine (Brian's university laptop) or blue crystal identified." 
fi

set -x

# The redis server is started on the first node.
REDIS_URL=redis://$SERVER_HOST:6379
echo "REDIS_URL is $REDIS_URL"
#TODO create a redis config

cd $lib_dir

mpirun -np 6 rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &
echo "launching rq worker"
#rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &
sleep 4

cd $script_dir
python3 Exp.py -rq=1 -p=150 -e=50 -bt=25 -host=$SERVER_HOST



sleep 1
echo "   SHUTDOWN REDIS   "
redis-cli shutdown

