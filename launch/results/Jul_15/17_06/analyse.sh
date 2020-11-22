
cd /home/bf16951/QMD/Launch/Results/Jul_15/17_06/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_06/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/17_06/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=118917ad87d452f5292b13abcd9bdaebf3984ded     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/17_06//bayes_factors.csv 

