#!/bin/bash

# set SERVER = $1
SERVER=$1
# cd $HOME/QMD/Libraries/QML_lib
echo "rq worker -u $SERVER"

rq worker -u $SERVER >> logs/worker_$HOSTNAME.log 2>&1
