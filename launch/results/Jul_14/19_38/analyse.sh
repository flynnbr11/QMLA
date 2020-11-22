
cd /home/bf16951/QMD/Launch/Results/Jul_14/19_38/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/19_38/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/19_38/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=1509c62536fbb9c3fbf3eff318b0c11ad0a821ad     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_14/19_38//bayes_factors.csv 

