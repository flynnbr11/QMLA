
cd /home/bf16951/QMD/Launch/Results/Sep_10/11_30/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/11_30/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/11_30/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=16114c70e75e77c67b20db3e6003a6d92cc27636     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_10/11_30//all_models_bayes_factors.csv 

