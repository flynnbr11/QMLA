
cd /home/bf16951/QMD/Launch/Results/Jul_16/14_27/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/14_27/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/14_27/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3826242cf89e303d7b7b2d0041406594dfe38783     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/14_27//bayes_factors.csv 

