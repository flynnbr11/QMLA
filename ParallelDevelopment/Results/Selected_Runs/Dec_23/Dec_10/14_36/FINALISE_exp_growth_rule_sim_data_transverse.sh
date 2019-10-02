
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/cumulative.csv -top=3 -qhl=0 -fqhl=0 -data=NVB_rescale_dataset.p -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/plot_probes.p 	-exp=0 -meas=hahn -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_expec_vals.p -latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/LatexMapping.txt -ggr=two_qubit_ising_rotation_hyperfine_transverse


qmd_id=100
cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment
for i in `seq 0 99`;
do
	let qmd_id=1+$qmd_id
	qsub -v QMD_ID=$qmd_id,OP=xTxTTiPPPiTxTTx,QHL=0,FURTHER_QHL=1,EXP_DATA=0,MEAS=hahn,GLOBAL_SERVER=newblue2,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36,DATETIME=Dec_10/14_36,NUM_PARTICLES=2500,NUM_EXP=750,NUM_BAYES=749,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=2.0,PLOTS=0,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/cumulative.csv,CUSTOM_PRIOR=1,DATASET=NVB_rescale_dataset.p,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205,GROWTH=two_qubit_ising_rotation_hyperfine_transverse,LATEX_MAP_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/LatexMapping.txt,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/prior.p,TRUE_EXPEC_PATH=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_expec_vals.p,PLOT_PROBES=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/plot_probes.p,RESOURCE_REALLOCATION=0,GAUSSIAN=1,PARAM_MIN=0,PARAM_MAX=8,PARAM_MEAN=0.5,PARAM_SIGMA=0.3 -N finalise_exp_growth_rule_sim_data_transverse\_$qmd_id -l walltime=00:00:4481,nodes=1:ppn=3 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/output_and_error_logs//finalise_error.txt run_qmd_instance.sh 
done 
	
