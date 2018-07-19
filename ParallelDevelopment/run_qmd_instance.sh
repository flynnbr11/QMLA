#!/bin/bash
#PBS -l nodes=1:ppn=3,walltime=00:06:00

rm dump.rdb 

let NUM_WORKERS="$PBS_NUM_NODES * $PBS_NUM_PPN"
let REDIS_PORT="6300 + $QMD_ID"
echo "QMD ID =$QMD_ID; REDIS_PORT=$REDIS_PORT"
echo "Global server: $GLOBAL_SERVER"
host=$(hostname)

if [ "$host" == "IT067176" ]
then
    echo "Brian's laptop identified"
    running_dir=$(pwd)
    lib_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/Libraries/QML_lib"
    script_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations"
    SERVER_HOST='localhost'
        
elif [[ "$host" == "newblue"* ]]
then
    echo "BC frontend identified"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load mvapich/gcc/64/1.2.0-qlc
    echo "launching redis"
    SERVER_HOST='localhost'

elif [[ "$host" == "node"* ]]
then
    echo "BC backend identified"
	running_dir="/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment"
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load languages/intel-compiler-16-u2
	
    SERVER_HOST=$(head -1 "$PBS_NODEFILE")
	cd $lib_dir
	redis-server RedisDatabaseConfig.conf --protected-mode no --port $REDIS_PORT & 
	redis-cli -p $REDIS_PORT flushall
else
    echo "Neither local machine (Brian's university laptop) or blue crystal identified." 
fi

cd $lib_dir
echo "Going in to launch redis script"
echo "If this fails -- ensure permission enabled on RedisLaunch script in library"

sleep 7

set -x
job_id=$PBS_JOBID

job_number="$(cut -d'.' -f1 <<<"$job_id")"
echo "Job id is $job_number"
cd $running_dir
mkdir -p $running_dir/logs
mkdir -p $PBS_O_WORKDIR/logs

if [ "$QHL" == 1 ]
then
	true_hamiltonian='z'
else
	true_hamiltonian='xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
fi


echo "Nodelist"
cat $confile 

# The redis server is started on the first node.
REDIS_URL=redis://$SERVER_HOST:$REDIS_PORT
echo "REDIS_URL is $REDIS_URL"
#TODO create a redis config

echo "Running dir is $running_dir"
echo "workers will log in $running_dir/logs"

#QMD_LOG_DIR="$PBS_O_WORKDIR/logs/$DATETIME"
QMD_LOG_DIR="$RESULTS_DIR/logs"
mkdir -p $QMD_LOG_DIR
QMD_JOB=$PBS_JOBNAME
echo "PBS job name is $QMD_JOB"
QMD_LOG="$QMD_LOG_DIR/$QMD_JOB.qmd.$job_number.log"


# Create the node file ---------------
# 
cat $PBS_NODEFILE
export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile=$QMD_LOG_DIR/node_info.$QMD_JOB.conf
for i in $nodes; do
 echo ${i} >>$confile
done
# -------------------------------------


cd $lib_dir
if [[ "$host" == "node"* ]]
then
	echo "Launching RQ worker on remote nodes using mpirun"
	mpirun -np $NUM_WORKERS -machinefile $confile rq worker $QMD_ID -u $REDIS_URL >> $QMD_LOG_DIR/$QMD_JOB.worker.$job_number.log 2>&1 &
else
	echo "Launching RQ worker locally"
	echo "RQ launched on $REDIS_URL at $(date +%H:%M:%S)" > $running_dir/logs/worker_$HOSTNAME.log 2>&1 
	rq worker -u $REDIS_URL >> $QMD_LOG_DIR/worker_$QMD_JOB.$HOSTNAME.log 2>&1 &
fi

sleep 5
cd $script_dir
echo "Starting Exp.py at $(date +%H:%M:%S); results dir: $RESULTS_DIR"

echo "CONFIG: -p=$NUM_PARTICLES -e=$NUM_EXP -bt=$NUM_BAYES -rt=$RESAMPLE_T -ra=$RESAMPLE_A -pgh=$RESAMPLE_PGH -qid=$QMD_ID -rqt=10000 -pkl=0 -host=$SERVER_HOST -port=$REDIS_PORT -dir=$RESULTS_DIR -log=$QMD_LOG"

python3 Exp.py -rq=1 -op=$true_hamiltonian -p=$NUM_PARTICLES -e=$NUM_EXP -bt=$NUM_BAYES -rt=$RESAMPLE_T -ra=$RESAMPLE_A -pgh=$RESAMPLE_PGH -qid=$QMD_ID -rqt=10000 -g=1 -exp=0 -qhl=$QHL -pt=$PLOTS -pkl=$PICKLE_QMD -host=$SERVER_HOST -port=$REDIS_PORT -dir=$RESULTS_DIR -log=$QMD_LOG -cb=$BAYES_CSV



echo "Finished Exp.py at $(date +%H:%M:%S); results dir: $RESULTS_DIR"
sleep 1

redis-cli -p $REDIS_PORT flushall
redis-cli -p $REDIS_PORT shutdown
echo "QMD $QMD_ID Finished; end of script at Time: $(date +%H:%M:%S)"
