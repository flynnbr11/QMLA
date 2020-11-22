#!/bin/bash

./local_launch.sh

cd $(ls -td -- ~/QMD/*/*/*/* | head -n 1)

./analyse.sh

num_pngs=$(ls *png | wc -l)
echo "$num_pngs PNGs generated."

cd ~/QMD/Launch
