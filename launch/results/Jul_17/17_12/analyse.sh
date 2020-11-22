
cd /home/bf16951/QMD/Launch/Results/Jul_17/17_12/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_17/17_12/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_17/17_12/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=8dabd64c784f8b881623a23f849ba39f61b19720     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_17/17_12//bayes_factors.csv 

