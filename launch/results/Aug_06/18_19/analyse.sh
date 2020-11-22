
cd /home/bf16951/QMD/Launch/Results/Aug_06/18_19/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_06/18_19/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_06/18_19/     -p=100     -e=25     -log=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=10c89b11f355c32ef6cfc0f42bc7e86045d4190d     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_06/18_19//bayes_factors.csv 

