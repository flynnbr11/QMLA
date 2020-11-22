
cd /home/bf16951/QMD/Launch/Results/Jul_14/19_07/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/19_07/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/19_07/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=1509c62536fbb9c3fbf3eff318b0c11ad0a821ad     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_14/19_07//bayes_factors.csv 

