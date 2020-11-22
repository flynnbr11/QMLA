
cd /home/bf16951/QMD/Launch/Results/Sep_30/22_36/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/22_36/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_30/22_36/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=2830d2a0f2d4fad485e0088c4c4aaebe082e8465     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_30/22_36//all_models_bayes_factors.csv 

