
cd /home/bf16951/QMD/Launch/Results/Jul_15/15_14/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/15_14/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/15_14/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=8fb31a1ab496448391644595f5a1a0a0995c664c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/15_14//bayes_factors.csv 

