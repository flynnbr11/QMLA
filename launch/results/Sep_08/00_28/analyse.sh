
cd /home/bf16951/QMD/Launch/Results/Sep_08/00_28/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/00_28/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/00_28/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=69792d5459f36b651ce9d590ea1e88f8368fd689     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_08/00_28//all_models_bayes_factors.csv 

