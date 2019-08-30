
	#!/bin/bash 
	cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
	python3 AnalyseMultipleQMD.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18 		--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/cumulative.csv 		-top=3 
		-qhl=0 		-fqhl=1 		-exp=1 		-latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/LatexMapping.txt 		-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/true_params.p 		-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/true_expec_vals.p 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/plot_probes.p

	python3 CombineAnalysisPlots.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18 		-p=1000 -e=250 -bt=249 -t=10 		-nprobes=40 		-pnoise=0.001 		-special_probe=dec_13_exp 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-run_desc=experimental_run_for_single_instance_tree 		-git_commit=2a397a89cb150fd91b1723f3444f9ef7e8ed304f 		-ra=0.98 		-rt=0.5 		-pgh=1.0 		-qhl=0 		-mqhl=0 		-cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_30/11_18/cumulative.csv 		-exp=1 		-out=further_qhl_analysis

