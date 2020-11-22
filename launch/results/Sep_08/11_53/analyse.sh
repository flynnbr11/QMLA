
cd /home/bf16951/QMD/Launch/Results/Sep_08/11_53/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/11_53/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//system_measurements.p     -ggr=NVPrelearnedTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/11_53/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//qmla.log     -ggr=NVPrelearnedTest     -run_desc="localdevelopemt"     -git_commit=3f792aa1f9534b08dfc4ec2911f7bc825e1437ff     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_08/11_53//all_models_bayes_factors.csv 

