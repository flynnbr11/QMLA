
cd /home/bf16951/QMD/Launch/Results/Sep_08/11_03/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/11_03/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_08/11_03/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=7b54141b5df0b0fd8227c1fa9ea03d6cde0acaf0     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_08/11_03//all_models_bayes_factors.csv 

