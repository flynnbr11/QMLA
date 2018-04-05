#!/bin/bash

qmd
cd Libraries/QML_Lib/
redis-server --protected-mode no &

./rq_worker_launch.sh localhost & 

qmd
cd ValidateQLE
python3 ExperimentalSpawningRule.py 

