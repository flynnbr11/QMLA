
	#!/bin/bash 
	cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
	python3 AnalyseMultipleQMD.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00 		--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/cumulative.csv 		-top=3 
		-qhl=0 		-fqhl=1 		-exp=1 		-latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/LatexMapping.txt 		-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/true_params.p 		-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/true_expec_vals.p 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/plot_probes.p

	python3 CombineAnalysisPlots.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00 		-p=2500 -e=750 -bt=749 -t=115 		-nprobes=40 		-pnoise=0.001 		-special_probe=dec_13_exp 		-ggr=two_qubit_ising_rotation_hyperfine_transverse 		-run_desc=simulate-nv-experiment__+R-probe 		-git_commit=f727f88be9318ac743b9d3fd5e3f1b85f144a69f 		-ra=0.98 		-rt=0.5 		-pgh=1.0 		-qhl=0 		-mqhl=0 		-cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Aug_16/18_00/cumulative.csv 		-exp=1 		-out=further_qhl_analysis

