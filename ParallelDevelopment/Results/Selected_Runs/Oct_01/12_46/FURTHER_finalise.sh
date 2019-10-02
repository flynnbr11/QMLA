
	#!/bin/bash 
	cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
	python3 AnalyseMultipleQMD.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46 		-log=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/qmd_log.log 		--bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/cumulative.csv 		-top=3 
		-qhl=0 		-fqhl=1 		-exp=0 		-latex==/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/LatexMapping.txt 		-params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/true_params.p 		-true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/true_expec_vals.p 		-ggr=NV_alternative_model 		-plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/plot_probes.p

	python3 CombineAnalysisPlots.py 		-dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46 		-p=2500 -e=750 -bt=749 -t=102 		-nprobes=40 		-pnoise=0.001 		-special_probe=random 		-ggr=NV_alternative_model 		-run_desc=NV-exp-method-sim__random--probe__QMLA 		-git_commit=38cb8a86c4097fa64e78884927ec04e306f56605 		-ra=0.98 		-rt=0.5 		-pgh=1.0 		-qhl=0 		-mqhl=0 		-cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_46/cumulative.csv 		-exp=0 		-out=further_qhl_analysis

