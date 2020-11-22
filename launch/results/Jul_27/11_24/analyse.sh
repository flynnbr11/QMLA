
cd /home/bf16951/QMD/Launch/Results/Jul_27/11_24/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_27/11_24/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_27/11_24/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=9493edde06d974ea63196fd4aa170aa73eb62c9b     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_27/11_24//bayes_factors.csv 

