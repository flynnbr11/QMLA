#!/bin/bash

# set SERVER = $1
SERVER=$1
# cd $HOME/QMD/Libraries/QML_lib
echo "rq worker -u redis://$SERVER:6379/1"

rq worker -u redis://$SERVER:6379/1 >> logs/worker_$HOSTNAME.log 2>&1
