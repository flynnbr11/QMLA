
cd /home/bf16951/QMD/Launch/Results/Jul_16/13_30/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/13_30/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/13_30/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=ae680400e9f8ab1416f19b71bed52e79eb32a3e2     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/13_30//bayes_factors.csv 

