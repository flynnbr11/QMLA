
cd /home/bf16951/QMD/Launch/Results/Jul_16/15_29/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/15_29/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/15_29/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3826242cf89e303d7b7b2d0041406594dfe38783     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/15_29//bayes_factors.csv 

