
cd /home/bf16951/QMD/Launch/Results/Jul_15/17_17/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_17/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_17/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=dbd42f2e99fef3807e5e580dc3b0d374c210b34e     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/17_17//bayes_factors.csv 

