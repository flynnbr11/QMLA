
cd /home/bf16951/QMD/Launch/Results/Jul_28/11_38/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/11_38/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/11_38/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=a68c16426a5cd61d2f1329fc36bb96d03ce02de0     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_28/11_38//bayes_factors.csv 

