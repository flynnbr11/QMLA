
cd /home/bf16951/QMD/Launch/Results/Jul_30/11_33/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/11_33/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/11_33/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=7547507df3105829768409027ee5a9908774c200     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_30/11_33//bayes_factors.csv 

