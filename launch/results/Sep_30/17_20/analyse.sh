
cd /home/bf16951/QMD/Launch/Results/Sep_30/17_20/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/17_20/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/17_20/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=021be525c9768948a0cf509fb2d7571efe4e0a80     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_30/17_20//all_models_bayes_factors.csv 

