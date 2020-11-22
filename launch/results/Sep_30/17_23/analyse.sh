
cd /home/bf16951/QMD/Launch/Results/Sep_30/17_23/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/17_23/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//system_measurements.p     -ggr=IsingGeneticTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/17_23/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//qmla.log     -ggr=IsingGeneticTest     -run_desc="localdevelopemt"     -git_commit=021be525c9768948a0cf509fb2d7571efe4e0a80     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_30/17_23//all_models_bayes_factors.csv 

