
cd /home/bf16951/QMD/Launch/Results/Jul_15/11_45/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/11_45/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/11_45/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=c1dacd5cff113d7298a5d0c77fb35d0b314facd4     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/11_45//bayes_factors.csv 

