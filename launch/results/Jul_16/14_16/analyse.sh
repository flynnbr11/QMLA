
cd /home/bf16951/QMD/Launch/Results/Jul_16/14_16/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/14_16/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/14_16/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=d96686ed58f4253a92c54f7b66add8dbe21454ad     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/14_16//bayes_factors.csv 

