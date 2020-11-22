
cd /home/bf16951/QMD/Launch/Results/Sep_05/11_37/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_05/11_37/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=3     -dir=/home/bf16951/QMD/Launch/Results/Sep_05/11_37/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=00620f241f7e39e428bc343a744e0e6c305c7e50     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_05/11_37//all_models_bayes_factors.csv 

