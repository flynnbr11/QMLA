#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=00:00:30

echo "Launching redis on $(hostname)"

module load tools/redis-4.0.8
nohup redis-server --protected-mode no & 

sleep 3
