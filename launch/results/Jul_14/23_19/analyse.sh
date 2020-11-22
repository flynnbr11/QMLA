
cd /home/bf16951/QMD/Launch/Results/Jul_14/23_19/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/23_19/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_14/23_19/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=10f407db505477742d2f103b9491cef26cfb6e49     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_14/23_19//bayes_factors.csv 

