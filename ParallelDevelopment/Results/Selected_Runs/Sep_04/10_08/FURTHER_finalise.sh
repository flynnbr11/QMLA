
	#!/bin/bash 
	cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
	python3 AnalyseMultipleQMD.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08 		--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/cumulative.csv 		-top=3 
		-qhl=0 		-fqhl=1 		-exp=1 		-latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/LatexMapping.txt 		-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/true_params.p 		-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/true_expec_vals.p 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/plot_probes.p

	python3 CombineAnalysisPlots.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08 		-p=25 -e=7 -bt=6 -t=3 		-nprobes=40 		-pnoise=0.001 		-special_probe=dec_13_exp 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-run_desc=experimental-run__plur-random-probe__test 		-git_commit=729f5eeab5e1afa355b9406140952ccec0958af6 		-ra=0.98 		-rt=0.5 		-pgh=1.0 		-qhl=0 		-mqhl=0 		-cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_04/10_08/cumulative.csv 		-exp=1 		-out=further_qhl_analysis

