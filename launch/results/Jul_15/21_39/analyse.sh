
cd /home/bf16951/QMD/Launch/Results/Jul_15/21_39/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/21_39/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/21_39/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=7e04ff11605d8c18e63ab95c881dd58913ce7976     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/21_39//bayes_factors.csv 

