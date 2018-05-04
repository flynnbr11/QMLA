#!/bin/bash

echo "GLobal host is $(hostname)" 

module load tools/redis-4.0.8
redis-cli flushall
redis-cli shutdown
redis-server --protected-mode no  & 

