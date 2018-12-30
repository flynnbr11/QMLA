
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42/cumulative.csv -top=3 -qhl=0 -fqhl=0 -data=NVB_rescale_dataset.p -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42/plot_probes.p 	-exp=0 -meas=hahn -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42/true_expec_vals.p -latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_23/19_42/LatexMapping.txt -ggr=two_qubit_ising_rotation_hyperfine_transverse

