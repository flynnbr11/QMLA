
cd /home/bf16951/QMD/Launch/Results/Jul_15/13_45/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/13_45/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/13_45/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=813d21af7d10b333ea2ba43c4878f7883e190601     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/13_45//bayes_factors.csv 

