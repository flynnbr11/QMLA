
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py 	-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05 	--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/cumulative.csv 	-top=3 	-qhl=0 	-fqhl=0 	-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/plot_probes.p 	-exp=0 	-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/true_params.p 	-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/true_expec_vals.p 	-latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/LatexMapping.txt 	-ggr=NV_alternative_model

python3 CombineAnalysisPlots.py     -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05     -p=2500 -e=750 -bt=750 -t=50     -nprobes=40     -pnoise=0.001     -special_probe=random     -ggr=NV_alternative_model     -run_desc=NV-exp-method-sim__probe-with-phase__different-probe__reduced-true-mod     -git_commit=09f8ba8e250903c99a4701fd30dd09ce0e063322     -ra=0.98     -rt=0.5     -pgh=1.0     -qhl=0     -mqhl=0     -cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/cumulative.csv     -exp=0


do_further_qhl=0
qmd_id=51
cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment
if (( $do_further_qhl == 1 ))
then

	for i in `seq 1 50`;
	do
		let qmd_id=1+$qmd_id
		finalise_script=finalise_$qmd_id
		qsub -v QMD_ID=$qmd_id,QHL=0,FURTHER_QHL=1,EXP_DATA=0,RUNNING_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment,LIBRARY_DIR=/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib,SCRIPT_DIR=/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations,GLOBAL_SERVER=newblue1,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05,DATETIME=Sep_19/12_05,NUM_PARTICLES=2500,NUM_EXP=750,NUM_BAYES=749,RESAMPLE_A=0.98,RESAMPLE_T=0.5,RESAMPLE_PGH=1.0,PLOTS=0,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/cumulative.csv,CUSTOM_PRIOR=1,DATA_MAX_TIME=5,GROWTH=NV_alternative_model,LATEX_MAP_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/LatexMapping.txt,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/prior.p,TRUE_EXPEC_PATH=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/true_expec_vals.p,PLOT_PROBES=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/plot_probes.p,RESOURCE_REALLOCATION=0,UPDATER_FROM_PRIOR=0,GAUSSIAN=1,PARAM_MIN=0,PARAM_MAX=10,PARAM_MEAN=0.5,PARAM_SIGMA=2 -N $finalise_script -l walltime=00:00:19493,nodes=1:ppn=3 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_19/12_05/output_and_error_logs//finalise_error.txt run_qmd_instance.sh 
	done 
fi

