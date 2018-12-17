
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/multiQMD.csv -top=2 -qhl=0 -data=NVB_dataset.p -exp=0 -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/true_params.p


cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment

qsub -v QMD_ID=102,OP=xTiPPyTiPPzTiPPxTxPPyTyPPzTz,QHL=0,FURTHER_QHL=1,EXP_DATA=0,GLOBAL_SERVER=newblue2,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46,DATETIME=Sep_26/09_46,NUM_PARTICLES=3600,NUM_EXP=1200,NUM_BAYES=1199,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=2.0,PLOTS=1,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/multiQMD.csv,CUSTOM_PRIOR=1,DATASET=NVB_dataset.p,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205,GROWTH=two_qubit_ising_rotation_hyperfine_transverse,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/prior.p -N finalise_simulated_transverse_positive_params_only -l walltime=10:00:00,nodes=1:ppn=2 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_26/09_46/output_and_error_logs//finalise_error.txt run_qmd_instance.sh
