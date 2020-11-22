
cd /home/bf16951/QMD/Launch/Results/Nov_15/11_25/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_15/11_25/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//system_measurements.p     -ggr=ThesisLatticeDemo     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_15/11_25/     -p=100     -e=20     -log=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//qmla.log     -ggr=ThesisLatticeDemo     -run_desc="localdevelopemt"     -git_commit=d185502d8f7fcfa10187bd591e5ce0c1614efd66     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_15/11_25//all_models_bayes_factors.csv 

