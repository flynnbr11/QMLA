
cd /home/bf16951/QMD/Launch/Results/Sep_02/11_13/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_02/11_13/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_02/11_13/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=a75698501a031349749d91f2e3ecbff0537feec7     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_02/11_13//all_models_bayes_factors.csv 

