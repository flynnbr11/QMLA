
#!/bin/bash 
cd /home/bf16951/QMD/scripts
python3 analyse_qmla.py 	-dir=/home/bf16951/QMD/Launch/Results/Sep_10/17_42/ 	-log=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//qmla_log.log 	--bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//cumulative.csv 	-top=4 	-qhl=1 	-fqhl=0 	-plotprobes=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//plot_probes.p 	-runinfo=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//true_params.p 	-sysmeas=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//system_measurements.p 	-latex=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//LatexMapping.txt 	-gs=1 	-ggr=NVCentreNQubitBath

python3 generate_results_pdf.py 	-dir=/home/bf16951/QMD/Launch/Results/Sep_10/17_42/ 	-p=30 	-e=10 	-t=5 	-ggr=NVCentreNQubitBath 	-run_desc=NV_bath-GR__2-qubit-taget__all-models-in-stage-at-once__qhl 	-git_commit=a6ad8c13f97776f14b6a0de1395ec7b4487d9cdd 	-qhl=1 	-mqhl=0 	-cb=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//cumulative.csv 


do_further_qhl=0
qmla_id=5
cd /home/bf16951/QMD/Launch
if (( $do_further_qhl == 1 ))
then

	for i in `seq 0 4`;
	do
		let qmla_id=1+$qmla_id
		finalise_script=finalise_$qmla_id
		qsub -v qmla_ID=$qmla_id,QHL=0,FURTHER_QHL=1,EXP_DATA=,RUNNING_DIR=/home/bf16951/QMD/Launch,LIBRARY_DIR=/home/bf16951/QMD/qmla,SCRIPT_DIR=/home/bf16951/QMD/scripts,GLOBAL_SERVER=IT067176,RESULTS_DIR=/home/bf16951/QMD/Launch/Results/Sep_10/17_42/,DATETIME=Sep_10/17_42,NUM_PARTICLES=30,NUM_EXP=10,NUM_BAYES=9,RESAMPLE_A=,RESAMPLE_T=,RESAMPLE_PGH=,PLOTS=0,PICKLE_qmla=0,BAYES_CSV=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//cumulative.csv,CUSTOM_PRIOR=,DATA_MAX_TIME=,GROWTH=NVCentreNQubitBath,LATEX_MAP_FILE=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//LatexMapping.txt,TRUE_PARAMS_FILE=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//true_params.p,PRIOR_FILE=,TRUE_EXPEC_PATH=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//system_measurements.p,PLOT_PROBES=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//plot_probes.p,RESOURCE_REALLOCATION=,UPDATER_FROM_PRIOR=,GAUSSIAN=,PARAM_MIN=,PARAM_MAX=,PARAM_MEAN=,PARAM_SIGMA= -N $finalise_script -l walltime=00:00:1300,nodes=1:ppn=4 -o /home/bf16951/QMD/Launch/Results/Sep_10/17_42//output_and_error_logs//finalise_output.txt -e /home/bf16951/QMD/Launch/Results/Sep_10/17_42//output_and_error_logs//finalise_error.txt run_qmla_instance.sh 
	done 
fi

