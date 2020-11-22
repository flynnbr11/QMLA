
cd /home/bf16951/QMD/Launch/Results/Sep_09/11_50/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_09/11_50/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//system_measurements.p     -ggr=NVPrelearnedTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_09/11_50/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//qmla.log     -ggr=NVPrelearnedTest     -run_desc="localdevelopemt"     -git_commit=c45221d6177449f0649c2631a195d2ffa7a7f4ef     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_09/11_50//all_models_bayes_factors.csv 

