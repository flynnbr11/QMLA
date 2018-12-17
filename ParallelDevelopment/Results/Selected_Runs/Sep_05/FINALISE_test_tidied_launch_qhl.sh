
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52/multiQMD.csv -top=2


#!/bin/bash 

cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment

qsub -v QMD_ID=100,OP=xTiPPyTiPPzTiPPxTxPPyTyPPzTz,QHL=0,FURTHER_QHL=1,EXP_DATA=1,GLOBAL_SERVER=newblue1,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52,DATETIME=Sep_05/17_52,NUM_PARTICLES=4000,NUM_EXP=2000,NUM_BAYES=1999,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=0.8,PLOTS=1,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52/multiQMD.csv,CUSTOM_PRIOR=1,DATASET=NVB_HahnPeaks_Newdata,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205 -N test_tidied_launch_qhl_100 -l walltime=00:00:133266,nodes=1:ppn=2 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52/OUTPUT_AND_ERROR_FILES//output_file_100.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_05/17_52/OUTPUT_AND_ERROR_FILES//error_file_100.txt run_qmd_instance.sh

