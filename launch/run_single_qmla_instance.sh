#!/bin/bash
#PBS -l nodes=1:ppn=5,walltime=00:06:00

rm dump.rdb 

let NUM_WORKERS="$PBS_NUM_NODES * $PBS_NUM_PPN"

if (( "$MULTIPLE_GROWTH_RULES" == 0))
then
	ALT_GROWTH=""
fi

# Assumed to be running on backend -> load redis as module
module load tools/redis-4.0.8

SERVER_HOST=$(head -1 "$PBS_NODEFILE")
let REDIS_PORT="6300 + $QMLA_ID"

cd $LIBRARY_DIR
redis-server RedisDatabaseConfig.conf --protected-mode no --port $REDIS_PORT & 
redis-cli -p $REDIS_PORT flushall

sleep 4 # time for redis server to start
set -x

job_id=$PBS_JOBID
job_number="$(cut -d'.' -f1 <<<"$job_id")"
echo "$job_number" >> $RESULTS_DIR/job_ids_started.txt

cd $RUNNING_DIR
mkdir -p $RUNNING_DIR/logs # TODO remove?
mkdir -p $PBS_O_WORKDIR/logs #TODO remove?

# The redis server is started on the first node.
REDIS_URL=redis://$SERVER_HOST:$REDIS_PORT
INSTANCE_LOG_DIRECTORY="$RESULTS_DIR/logs"
OUTPUT_ERROR_DIR="$RESULTS_DIR/output_and_error_logs"
mkdir -p $INSTANCE_LOG_DIRECTORY
QMLA_JOB=$PBS_JOBNAME
echo "PBS job name is $QMLA_JOB"
QMLA_LOG="$INSTANCE_LOG_DIRECTORY/$QMLA_JOB.qmd.$job_number.log"


# Create the node file ---------------
# 
cat $PBS_NODEFILE
export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile="$OUTPUT_ERROR_DIR/node_info.$QMLA_JOB.conf"
for i in $nodes; do
	echo ${i} >>$confile
done
# -------------------------------------


# Launch RQ workers from QMLA root directory so that import statements calling qmla are understood
cd $ROOT_DIR

let NUM_RQ_WORKERS="$NUM_WORKERS-1"
set -x 
mpirun --display-map --tag-output \
	-np 1 \
	python3 scripts/implement_qmla.py \
	-qid=$QMLA_ID \
	-dir=$RESULTS_DIR \
	-rq=1 \
	-qhl=$RUN_QHL \
	-fq=$FURTHER_QHL \
	-mqhl=$RUN_QHL_MULTI_MODEL \
	-e=$NUM_EXPERIMENTS \
	-p=$NUM_PARTICLES \
	-host=$SERVER_HOST \
	-port=$REDIS_PORT \
	-pkl=$PICKLE_INSTANCE \
	-pt=$PLOTS \
	-cb=$BAYES_CSV \
	-log=$QMLA_LOG \
	-runinfo=$RUN_INFO_FILE \
	-sysmeas=$SYS_MEAS_FILE \
	-plotprobes=$PLOT_PROBES_FILE \
	-latex=$LATEX_MAP_FILE \
	-pl=$PLOT_LEVEL \
	-debug=$DEBUG \
	-ggr=$GROWTH_RULE \
	$ALT_GROWTH \
	> $RESULTS_DIR/output_and_error_logs/profile_$QMLA_ID.txt \
	: \
	-np $NUM_RQ_WORKERS \
	python3 rq_worker_qmla.py -host=$SERVER_HOST -port=$REDIS_PORT -qid=$QMLA_ID >> $INSTANCE_LOG_DIRECTORY/$QMLA_JOB.worker.$job_number.log 2>&1 


echo "$job_number" >> $RESULTS_DIR/job_ids_completed.txt
echo "QMLA instace $QMLA_ID finished at time: $(date +%H:%M:%S)"
