
	#!/bin/bash 
	cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
	python3 AnalyseMultipleQMD.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42 		--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/cumulative.csv 		-top=3 
		-qhl=0 		-fqhl=1 		-exp=1 		-latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/LatexMapping.txt 		-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/true_params.p 		-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/true_expec_vals.p 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/plot_probes.p

	python3 CombineAnalysisPlots.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42 		-p=2500 -e=750 -bt=749 -t=50 		-nprobes=40 		-pnoise=0.001 		-special_probe=dec_13_exp 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-run_desc=NV-exp__probe-with-phase__QHL 		-git_commit=aaff7013310476af36d4c04910f5b58f0598c1d0 		-ra=0.98 		-rt=0.5 		-pgh=1.0 		-qhl=0 		-mqhl=0 		-cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_16/17_42/cumulative.csv 		-exp=1 		-out=further_qhl_analysis

