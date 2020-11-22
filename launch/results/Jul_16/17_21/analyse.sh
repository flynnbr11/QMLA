
cd /home/bf16951/QMD/Launch/Results/Jul_16/17_21/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/17_21/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_16/17_21/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=c282ebd8f9b27af3d1e2855c500725a83ea251bb     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_16/17_21//bayes_factors.csv 

