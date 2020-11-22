
cd /home/bf16951/QMD/Launch/Results/Jul_27/13_24/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_27/13_24/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_27/13_24/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=da285972a4ccd75c7172b23044ea0887ce8236dc     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_27/13_24//bayes_factors.csv 

