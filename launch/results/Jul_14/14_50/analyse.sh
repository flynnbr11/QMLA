
cd /home/bf16951/QMD/Launch/Results/Jul_14/14_50/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/14_50/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/14_50/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=ca2346ab0943f4efac523387b6f7b503813451b1     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_14/14_50//bayes_factors.csv 

