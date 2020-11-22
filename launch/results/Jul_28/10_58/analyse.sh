
cd /home/bf16951/QMD/Launch/Results/Jul_28/10_58/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/10_58/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_28/10_58/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3545c87ab467bae3c8a644287d09c271f35c816c     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_28/10_58//bayes_factors.csv 

