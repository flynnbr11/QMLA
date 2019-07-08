
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py 	-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40 	--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/cumulative.csv 	-top=3 	-qhl=0 	-fqhl=0 	-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/plot_probes.p 	-exp=1 	-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/true_params.p 	-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/true_expec_vals.p 	-latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/LatexMapping.txt 	-ggr=two_qubit_ising_rotation_hyperfine_transverse

python3 CombineAnalysisPlots.py     -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40     -p=2500 -e=750 -bt=750 -t=115     -nprobes=40     -pnoise=0.001     -special_probe=dec_13_exp     -ggr=two_qubit_ising_rotation_hyperfine_transverse     -run_desc=exp-paper__experimental-data     -git_commit=54b1cf14f77e235d3bd304f7121265827a8f0760     -ra=0.98     -rt=0.5     -pgh=1.0     -qhl=0     -mqhl=0     -cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/cumulative.csv     -exp=1


do_further_qhl=0
qmd_id=115
cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment
if (( $do_further_qhl == 1 ))
then

	for i in `seq 0 114`;
	do
		let qmd_id=1+$qmd_id
		finalise_script=finalise_$qmd_id
		qsub -v QMD_ID=$qmd_id,QHL=0,FURTHER_QHL=1,EXP_DATA=1,RUNNING_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment,LIBRARY_DIR=/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib,SCRIPT_DIR=/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations,GLOBAL_SERVER=newblue1,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40,DATETIME=Jul_05/16_40,NUM_PARTICLES=2500,NUM_EXP=750,NUM_BAYES=749,RESAMPLE_A=0.98,RESAMPLE_T=0.5,RESAMPLE_PGH=1.0,PLOTS=0,PICKLE_QMD=1,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/cumulative.csv,CUSTOM_PRIOR=1,DATA_MAX_TIME=15,GROWTH=two_qubit_ising_rotation_hyperfine_transverse,LATEX_MAP_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/LatexMapping.txt,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/prior.p,TRUE_EXPEC_PATH=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/true_expec_vals.p,PLOT_PROBES=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/plot_probes.p,RESOURCE_REALLOCATION=0,UPDATER_FROM_PRIOR=0,GAUSSIAN=1,PARAM_MIN=0,PARAM_MAX=10,PARAM_MEAN=0.5,PARAM_SIGMA=2 -N $finalise_script -l walltime=00:00:19493,nodes=1:ppn=3 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jul_05/16_40/output_and_error_logs//finalise_error.txt run_qmd_instance.sh 
	done 
fi

