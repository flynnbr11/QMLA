
cd /home/bf16951/QMD/Launch/Results/Jul_15/17_02/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_02/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_02/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=118917ad87d452f5292b13abcd9bdaebf3984ded     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/17_02//bayes_factors.csv 

