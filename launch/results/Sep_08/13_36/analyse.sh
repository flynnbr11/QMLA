
cd /home/bf16951/QMD/Launch/Results/Sep_08/13_36/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/13_36/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//system_measurements.p     -ggr=NVPrelearnedTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/13_36/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//qmla.log     -ggr=NVPrelearnedTest     -run_desc="localdevelopemt"     -git_commit=5c6701d7e03993591d94b3e76b21d0b1ab1a699a     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_08/13_36//all_models_bayes_factors.csv 

