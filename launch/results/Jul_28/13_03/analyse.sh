
cd /home/bf16951/QMD/Launch/Results/Jul_28/13_03/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/13_03/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/13_03/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=0326f63d8b5f9446510c72d7a4aab552fce23f17     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_28/13_03//bayes_factors.csv 

