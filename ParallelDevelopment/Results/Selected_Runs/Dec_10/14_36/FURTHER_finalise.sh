
		#!/bin/bash 
		cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
		python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/cumulative.csv -top=3 -qhl=0 -fqhl=1 -data=NVB_rescale_dataset.p -exp=0 -meas=hahn -latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/LatexMapping.txt -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/true_expec_vals.p -ggr=two_qubit_ising_rotation_hyperfine_transverse -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_10/14_36/plot_probes.p
	
