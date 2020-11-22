
cd /home/bf16951/QMD/Launch/Results/Nov_18/21_26/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_18/21_26/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_18/21_26/     -p=20     -e=10     -log=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=13a7d98ced6e56e4668f5c32e2bee9b20e8cfb1b     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_18/21_26//all_models_bayes_factors.csv 

