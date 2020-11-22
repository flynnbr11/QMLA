
	#!/bin/bash 
	cd /home/bf16951/QMD/scripts
	python3 analyse_qmla.py 		-dir=/home/bf16951/QMD/Launch/Results/Sep_10/17_42/ 		-log=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//qmla_log.log 		--bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//cumulative.csv 		-top=4 
		-qhl=1 		-fqhl=1 		-latex==/home/bf16951/QMD/Launch/Results/Sep_10/17_42//LatexMapping.txt 		-runinfo=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//true_params.p 		-sysmeas=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//system_measurements.p 		-ggr=NVCentreNQubitBath 		-plotprobes=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//plot_probes.p

	python3 generate_results_pdf.py 		-dir=/home/bf16951/QMD/Launch/Results/Sep_10/17_42/ 		-p=30 -e=10 -bt=9 -t=5 		-nprobes= 		-pnoise= 		-special_probe= 		-ggr=NVCentreNQubitBath 		-run_desc=NV_bath-GR__2-qubit-taget__all-models-in-stage-at-once__qhl 		-git_commit=a6ad8c13f97776f14b6a0de1395ec7b4487d9cdd 		-ra= 		-rt= 		-pgh= 		-qhl=1 		-mqhl=0 		-cb=/home/bf16951/QMD/Launch/Results/Sep_10/17_42//cumulative.csv 		-exp= 		-out=further_qhl_analysis

