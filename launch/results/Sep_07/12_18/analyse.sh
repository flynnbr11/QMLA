
cd /home/bf16951/QMD/Launch/Results/Sep_07/12_18/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_07/12_18/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_07/12_18/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=5c95dfa8e2ce989dbf69478156cf000d0b3962e3     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_07/12_18//all_models_bayes_factors.csv 

