
cd /home/bf16951/QMD/Launch/Results/Nov_16/23_30/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_16/23_30/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//system_measurements.p     -ggr=DemoIsing     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_16/23_30/     -p=30     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//qmla.log     -ggr=DemoIsing     -run_desc="localdevelopemt"     -git_commit=d35db3d69a520f3614dd063eaee4e0edb2d9ab55     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_16/23_30//all_models_bayes_factors.csv 

