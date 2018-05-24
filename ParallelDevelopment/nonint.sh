#!/bin/bash

num_tests=1
min_id=2
let max_id="$min_id + $num_tests - 1 "

p_min=100
p_max=2000
p_int=400

e_min=50 
e_max=200
e_int=50

ra_min=0.8
ra_max=0.8
ra_int=0.05

rt_min=0.35
rt_max=0.35
rt_int=0.05

rp_min=0.8
rp_max=0.8
rp_int=0.1


count=0


for e in `seq $e_min $e_int $e_max`;
do 
	for p in `seq $p_min $p_int $p_max `;
	do 

		for ra in `seq $ra_min $ra_int $ra_max `;
		do
			for rt in `seq $rt_min $rt_int $rt_max `;
			do
				for rp in `seq $rp_min $rp_int $rp_max`;
				do		

					for i in `seq $min_id $max_id`;
					do

						let count="$count+1"
						echo "p=$p; e=$e, ra=$ra; rt=$rt; rp=$rp;    count=$count"
					done
				done
			done
		done

	done
done

echo "\n\n Final Count: $count"
