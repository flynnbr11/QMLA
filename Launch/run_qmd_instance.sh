#!/bin/bash
#PBS -l nodes=1:ppn=5,walltime=00:06:00

print_passed_variables = 1
if (( "$print_passed_variables" == 1 ))
then
	echo "
	Inside run_qmd script. \m
	QMD_ID=$QMD_ID \n
	QHL=$QHL \n
	SCRIPT_DIR = $SCRIPT_DIR
	"
fi
rm dump.rdb 

let NUM_WORKERS="$PBS_NUM_NODES * $PBS_NUM_PPN"
let REDIS_PORT="6300 + $QMD_ID"
host=$(hostname)
running_dir=$RUNNING_DIR
lib_dir=$LIBRARY_DIR
script_dir=$SCRIPT_DIR
root_dir=$ROOT_DIR
echo "QMD ID = $QMD_ID; REDIS_PORT=$REDIS_PORT"
echo "Global server: $GLOBAL_SERVER"
echo "Running directory: $running_dir"
echo "Library directory: $lib_dir"
echo "Script directory: $script_dir"


# assumed to be running on backend, where redis is loaded as below
module load tools/redis-4.0.8
module load languages/intel-compiler-16-u2

SERVER_HOST=$(head -1 "$PBS_NODEFILE")
cd $lib_dir
#cd $script_dir
redis-server RedisDatabaseConfig.conf --protected-mode no --port $REDIS_PORT & 
redis-cli -p $REDIS_PORT flushall

#cd $lib_dir
echo "Going in to launch redis script"
echo "If this fails -- ensure permission enabled on RedisLaunch script in library"

sleep 4 # this might need to be increased?

set -x
job_id=$PBS_JOBID

job_number="$(cut -d'.' -f1 <<<"$job_id")"
echo "Job id is $job_number"
echo "$job_number" >> $RESULTS_DIR/job_ids_started.txt
cd $running_dir
mkdir -p $running_dir/logs
mkdir -p $PBS_O_WORKDIR/logs



# The redis server is started on the first node.
REDIS_URL=redis://$SERVER_HOST:$REDIS_PORT
echo "REDIS_URL : $REDIS_URL"
QMD_LOG_DIR="$RESULTS_DIR/logs"
OUTPUT_ERROR_DIR="$RESULTS_DIR/output_and_error_logs"
mkdir -p $QMD_LOG_DIR
QMD_JOB=$PBS_JOBNAME
echo "PBS job name is $QMD_JOB"
QMD_LOG="$QMD_LOG_DIR/$QMD_JOB.qmd.$job_number.log"


# Create the node file ---------------
# 
cat $PBS_NODEFILE
export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile="$OUTPUT_ERROR_DIR/node_info.$QMD_JOB.conf"
for i in $nodes; do
	echo ${i} >>$confile
done
# -------------------------------------


#cd $lib_dir
cd $root_dir
if [[ "$host" == "node"* ]]
then
	echo "Launching RQ worker on remote nodes using mpirun"
	mpirun -np $NUM_WORKERS -machinefile $confile rq worker $QMD_ID -u $REDIS_URL >> $QMD_LOG_DIR/$QMD_JOB.worker.$job_number.log 2>&1 &
else
	echo "Launching RQ worker locally"
	echo "RQ launched on $REDIS_URL at $(date +%H:%M:%S)" > $running_dir/logs/worker_$HOSTNAME.log 2>&1 
	rq worker -u $REDIS_URL >> $QMD_LOG_DIR/worker_$QMD_JOB.$HOSTNAME.log 2>&1 &
fi

printf "In RUN script: $ALT_GROWTH" > $QMD_LOG

if (( "$MULTIPLE_GROWTH_RULES" == 0))
then
	ALT_GROWTH=""
fi
sleep 5
# cd $running_dir
cd $script_dir
python3 implement_qmla.py -mqhl=$MULTIPLE_QHL -rq=1 -p=$NUM_PARTICLES -e=$NUM_EXP -bt=$NUM_BAYES -rt=$RESAMPLE_T -ra=$RESAMPLE_A -pgh=$RESAMPLE_PGH -pgh_exp=$PGH_EXPONENT -pgh_incr=$PGH_INCREASE -qid=$QMD_ID -rqt=200000 -g=$GAUSSIAN -exp=$EXP_DATA -qhl=$QHL -fq=$FURTHER_QHL -pt=$PLOTS -pkl=$PICKLE_QMD -host=$SERVER_HOST -port=$REDIS_PORT -dir=$RESULTS_DIR -log=$QMD_LOG -cb=$BAYES_CSV -cpr=$CUSTOM_PRIOR -prtwt=$STORE_PARTICLES_WEIGHTS -dst=$DATA_MAX_TIME -ggr=$GROWTH $ALT_GROWTH -bintimes=$BIN_TIMES_BAYES -bftimesall=$BF_ALL_TIMES -latex=$LATEX_MAP_FILE -nprobes=$NUM_PROBES -prior_path=$PRIOR_FILE -true_params_path=$TRUE_PARAMS_FILE -plot_probes=$PLOT_PROBES -special_probe=$SPECIAL_PROBE -pnoise=$PROBE_NOISE -true_expec_path=$TRUE_EXPEC_PATH -pmin=$PARAM_MIN -pmax=$PARAM_MAX -pmean=$PARAM_MEAN -psigma=$PARAM_SIGMA -resource=$RESOURCE_REALLOCATION --updater_from_prior=$UPDATER_FROM_PRIOR  



echo "Finished Exp.py at $(date +%H:%M:%S); results dir: $RESULTS_DIR"
sleep 1

redis-cli -p $REDIS_PORT flushall
redis-cli -p $REDIS_PORT shutdown
echo "$job_number" >> $RESULTS_DIR/job_ids_completed.txt
echo "QMD $QMD_ID Finished; end of script at Time: $(date +%H:%M:%S)"
