#!/bin/bash

host=$(hostname)


echo "Inside launch redis script; host=$host"

if [ "$host" == "IT067176" ]
then
    echo "Brian's laptop identified -  launching redis"
    running_dir=$(pwd)
    lib_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/Libraries/QML_lib"
    script_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations"
    SERVER_HOST='localhost'
    ~/redis-4.0.8/src/redis-server  $lib_dir/RedisConfig.conf & 
        
elif [[ "$host" == "newblue"* ]]
then
    echo "BC frontend identified"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load mvapich/gcc/64/1.2.0-qlc
    echo "launching redis"
    redis-server $lib_dir/RedisConfig.conf --protected-mode no  &
    SERVER_HOST='localhost'


elif [[ "$host" == "node"* ]]
then
    echo "BC backend identified"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load languages/intel-compiler-16-u2
#    SERVER_HOST=$(head -1 "$PBS_NODEFILE")
	SERVER_HOST=$(hostname)
    echo "launching redis: $lib_dir/RedisConfig.conf on $SERVER_HOST"
	cd $lib_dir    
	redis_run_test=`python3 RedisCheck.py -rh=$SERVER_HOST`
	redis_test=$(echo $redis_run_test)	
	echo "redis test: $redis_test"
	if [[ "$redis_test" == "redis-ready" ]]
	then
		echo "Redis server already present on $SERVER_HOST"
		echo "Time: $(date +%H:%M:%S)"
		python3 RedisManageServer.py -rh=$SERVER_HOST -rqid=$QMD_ID -action='add'
	else 
		echo "Redis server NOT already present on $SERVER_HOST; launching"
		redis-server RedisDatabaseConfig.conf --protected-mode no &
        redis-cli flushall
		python3 RedisManageServer.py -rh=$SERVER_HOST -rqid=$QMD_ID -action='add'


	fi
else
    echo "Neither local machine (Brian's university laptop) or blue crystal identified." 
fi


