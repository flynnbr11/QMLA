
cd /home/bf16951/QMD/Launch/Results/Nov_16/15_29/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_16/15_29/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//system_measurements.p     -ggr=IsingDemo     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_16/15_29/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//qmla.log     -ggr=IsingDemo     -run_desc="localdevelopemt"     -git_commit=d35db3d69a520f3614dd063eaee4e0edb2d9ab55     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_16/15_29//all_models_bayes_factors.csv 

