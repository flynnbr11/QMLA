#!/bin/bash

cd Libraries/QML_lib

~/redis-4.0.8/src/redis-server --protected-mode no &
# mpirun -np 1 -ppn 4 ./rq_worker_launch.sh localhost & 
# rq worker -u redis://localhost:6379/1 & 

HOST='127.0.0.1'
# HOST='localhost'
REDIS_URL=redis://$HOST:6379

echo "REDIS_URL is $REDIS_URL"

# mpirun -np 1 -ppn 8 rq worker -u $REDIS_URL &
mpirun -np 2 -ppn 2 rq worker -u $REDIS_URL > logs/worker_$HOSTNAME.log 2>&1 &


cd ../../ValidateQLE
python3 ExperimentalSpawningRule.py -rq=1 -host=$HOST



sleep 1
echo "   SHUTDOWN REDIS   "
redis-cli shutdown

