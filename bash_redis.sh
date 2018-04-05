#!/bin/bash

cd Libraries/QML_lib

~/redis-4.0.8/src/redis-server --protected-mode no &
# mpirun -np 1 -ppn 4 ./rq_worker_launch.sh localhost & 
rq worker -u redis://localhost:6379/1

cd ../../ValidateQLE
python3 ExperimentalSpawningRule.py 


sleep 5
echo "   SHUTDOWN REDIS   "
redis-cli shutdown

