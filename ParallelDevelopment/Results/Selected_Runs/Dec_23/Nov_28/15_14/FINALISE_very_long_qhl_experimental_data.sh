
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14/cumulative.csv -top=3 -qhl=1 -fqhl=0 -data=NVB_rescale_dataset.p -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14/plot_probes.p 	-exp=1 -meas=hahn -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14/true_expec_vals.p -latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Nov_28/15_14/LatexMapping.txt -ggr=two_qubit_ising_rotation_hyperfine_transverse

