
cd /home/bf16951/QMD/Launch/Results/Jul_16/15_59/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/15_59/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/15_59/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=a0490adfb882ac5e5c4026e266e90588c292e0fe     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/15_59//bayes_factors.csv 

