
cd /home/bf16951/QMD/Launch/Results/Jul_14/21_12/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/21_12/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/21_12/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=21408c5e978b2b8c3632e5361f3a29e96952641e     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_14/21_12//bayes_factors.csv 

