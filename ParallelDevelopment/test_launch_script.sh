#!/bin/bash

test_description="wider_initial_dist"

OUT_LOG="$(pwd)/logs/OUTPUT_AND_ERROR_FILES"
echo "pwd: $(pwd)"
echo "OUT LOG: $OUT_LOG"
mkdir -p $OUT_LOG
output_file="output_file"
error_file="error_file" 

num_tests=3
min_id=1
let max_id="$min_id + $num_tests - 1 "


this_dir=$(hostname)
day_time=$(date +%b_%d/%H_%M)

script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
results_dir=$day_time
full_path_to_results=$script_dir/Results/$results_dir
all_qmd_bayes_csv="$full_path_to_results/multiQMD.csv"


mkdir -p results_dir

global_server=$(hostname)

very_short_time="walltime=00:06:00"
short_time="walltime=00:20:00"
medium_time="walltime=01:00:00"
long_time="walltime=08:00:00"
very_long_time="walltime=16:00:00"

test_time="walltime=00:90:00"

time=$test_time
qmd_id=$min_id
cutoff_time=180

do_plots=1
pickle_class=0

p_min=3000
p_max=3000
p_int=1000
p_default=2000

e_min=2000 
e_max=2000
e_int=1000
e_default=2000

ra_min=0.8
ra_max=0.95
ra_int=0.05
ra_default=0.9

rt_min=0.4
rt_max=0.6
rt_int=0.1
rt_default=0.5

rp_min=0.9
rp_max=1.1
rp_int=0.1
rp_default=1.0


e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

e=5
p=10

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
#	  only need one hour while testing evo failure
#	time="walltime=01:00:00"
	this_qmd_name="$test_description""_$qmd_id"
	this_error_file="$OUT_LOG/$error_file""_$qmd_id.txt"
	this_output_file="$OUT_LOG/$output_file""_$qmd_id.txt"
	echo "This output file: $this_output_file"
	echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp,PLOTS=$do_plots,PICKLE_QMD=$pickle_class,BAYES_CSV=$all_qmd_bayes_csv -N $this_qmd_name -l $time -o $this_output_file -e $this_error_file launch_qmd_parallel.sh

done

: <<'END'
e=1000
p=1000

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
#	  only need one hour while testing evo failure
#	time="walltime=01:00:00"
	this_qmd_name="$test_description""_$qmd_id"
	echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh

done

e=2000
p=1000

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
#	  only need one hour while testing evo failure
#	time="walltime=01:00:00"
	this_qmd_name="$test_description""_$qmd_id"
	echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh

done

e=2000
p=200

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
#	  only need one hour while testing evo failure
#	time="walltime=01:00:00"
	this_qmd_name="$test_description""_$qmd_id"
	echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh

done


e=2000
p=2000

for i in `seq $min_id $max_id`;
do


	let bt="$e/2"
	let qmd_id="$qmd_id+1"
	let ham_exp="$e*$p + $p*$bt"
	let expected_time="$ham_exp/50"
	if (( $expected_time < $cutoff_time));
	then
		seconds_reqd=$cutoff_time	
	else
		seconds_reqd=$expected_time	
	fi
	time="walltime=00:00:$seconds_reqd"
#	  only need one hour while testing evo failure
#	time="walltime=01:00:00"
	this_qmd_name="$test_description""_$qmd_id"
	echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
	echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
	qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh

done
END



: <<'END'

for e in `seq $e_min $e_int $e_max`;
do 
	for p in `seq $p_min $p_int $p_max `;
	do 

		for i in `seq $min_id $max_id`;
		do
			let bt="$e/2"
			let qmd_id="$qmd_id+1"
			let ham_exp="$e*$p + $p*$bt"
			let expected_time="$ham_exp/50"
			if (( $expected_time < $cutoff_time));
			then
				seconds_reqd=$cutoff_time	
			else
				seconds_reqd=$expected_time	
			fi
			time="walltime=00:00:$seconds_reqd"
			this_qmd_name="$test_description""_$qmd_id"
			echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
			echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
			qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
		done 
	done
done





# Loop over resample_a
for ra in `seq $ra_min $ra_int $ra_max `;
do
	for i in `seq $min_id $max_id`;
	do
		let bt="$e/2"
		let qmd_id="$qmd_id+1"
		let ham_exp="$e*$p + $p*$bt"
		let expected_time="$ham_exp/50"
		if (( $expected_time < $cutoff_time));
		then
			seconds_reqd=$cutoff_time	
		else
			seconds_reqd=$expected_time	
		fi
		time="walltime=00:00:$seconds_reqd"
		this_qmd_name="$test_description""_$qmd_id"
		echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
		echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
		qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
	done 
done

e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

# Loop over resample_t
for rt in `seq $rt_min $rt_int $rt_max `;
do
	for i in `seq $min_id $max_id`;
	do
		let bt="$e/2"
		let qmd_id="$qmd_id+1"
		let ham_exp="$e*$p + $p*$bt"
		let expected_time="$ham_exp/50"
		if (( $expected_time < $cutoff_time));
		then
			seconds_reqd=$cutoff_time	
		else
			seconds_reqd=$expected_time	
		fi
		time="walltime=00:00:$seconds_reqd"
		this_qmd_name="$test_description""_$qmd_id"
		echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
		echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
		qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
	done 
done

e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

# Loop over resample_pgh_factor
for rp in `seq $rp_min $rp_int $rp_max`;
do		
	for i in `seq $min_id $max_id`;
	do
		let bt="$e/2"
		let qmd_id="$qmd_id+1"
		let ham_exp="$e*$p + $p*$bt"
		let expected_time="$ham_exp/50"
		if (( $expected_time < $cutoff_time));
		then
			seconds_reqd=$cutoff_time	
		else
			seconds_reqd=$expected_time	
		fi
		time="walltime=00:00:$seconds_reqd"
		this_qmd_name="$test_description""_$qmd_id"
		echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
		echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
		qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
	done 
done

e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

# Loop over experiment number
for e in `seq $e_min $e_int $e_max`;
do 
	for i in `seq $min_id $max_id`;
	do
		let bt="$e/2"
		let qmd_id="$qmd_id+1"
		let ham_exp="$e*$p + $p*$bt"
		let expected_time="$ham_exp/50"
		if (( $expected_time < $cutoff_time));
		then
			seconds_reqd=$cutoff_time	
		else
			seconds_reqd=$expected_time	
		fi
		time="walltime=00:00:$seconds_reqd"
		this_qmd_name="$test_description""_$qmd_id"
		echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
		echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
		qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
	done 
done

e=$e_default
p=$p_default
ra=$ra_default
rt=$rt_default
rp=$rp_default

# Loop over particle number
for p in `seq $p_min $p_int $p_max `;
do 

	for i in `seq $min_id $max_id`;
	do
		let bt="$e/2"
		let qmd_id="$qmd_id+1"
		let ham_exp="$e*$p + $p*$bt"
		let expected_time="$ham_exp/50"
		if (( $expected_time < $cutoff_time));
		then
			seconds_reqd=$cutoff_time	
		else
			seconds_reqd=$expected_time	
		fi
		time="walltime=00:00:$seconds_reqd"
		this_qmd_name="$test_description""_$qmd_id"
		echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
		echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
		qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
	done 
done

END



: <<'END'
## Complete loop

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
						let bt="$e/2"
						let qmd_id="$qmd_id+1"
						let ham_exp="$e*$p + $p*$bt"
						let seconds_reqd="$ham_exp/50"
						time="walltime=00:00:$seconds_reqd"
						this_qmd_name="$test_description""_$qmd_id"
						echo "QMD ID: $qmd_id \t num particles:$NUM_PARTICLES"
						echo "Config: e=$e; p=$p; bt=$bt; ra=$ra; rt=$rt; rp=$rp; qid=$qmd_id; seconds=$seconds_reqd"
						qsub -v QMD_ID=$qmd_id,GLOBAL_SERVER=$global_server,RESULTS_DIR=$results_dir,NUM_PARTICLES=$p,NUM_EXP=$e,NUM_BAYES=$bt,RESAMPLE_A=$ra,RESAMPLE_T=$rt,RESAMPLE_PGH=$rp -N $this_qmd_name -l $time launch_qmd_parallel.sh
					done 
				done
			done
		done
	done
done
END



echo "
#!/bin/bash 
cd ../Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir="$full_path_to_results" --bayes_csv=$all_qmd_bayes_csv
" > analyse_results/analyse_$test_description.sh
