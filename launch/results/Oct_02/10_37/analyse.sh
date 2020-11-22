
cd /home/bf16951/QMD/Launch/Results/Oct_02/10_37/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Oct_02/10_37/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Oct_02/10_37/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3ca67119f1c81183d4adae28e10fb577e5abbefe     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Oct_02/10_37//all_models_bayes_factors.csv 

