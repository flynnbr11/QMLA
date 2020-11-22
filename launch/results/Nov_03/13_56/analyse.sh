
cd /home/bf16951/QMD/Launch/Results/Nov_03/13_56/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_03/13_56/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//system_measurements.p     -ggr=GRTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_03/13_56/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//qmla.log     -ggr=GRTest     -run_desc="localdevelopemt"     -git_commit=64037f7c53d241918d5d5b86556ac64a7e63fa14     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_03/13_56//all_models_bayes_factors.csv 

