
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_11/13_53 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_11/13_53/multiQMD.csv -top=2 -qhl=1 -fqhl=0 -data=NVB_dataset.p -exp=0 -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_11/13_53/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_11/13_53/true_expec_vals.p

