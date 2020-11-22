
cd /home/bf16951/QMD/Launch/Results/Sep_08/00_07/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/00_07/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/00_07/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=9202e65afdfdade6bc86cca1e5b7ed16b95624cb     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_08/00_07//all_models_bayes_factors.csv 

