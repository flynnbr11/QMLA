
cd /home/bf16951/QMD/Launch/Results/Jul_10/14_49/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_10/14_49/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//system_measurements.p     -ggr=NVCentreNQubitBath     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_10/14_49/     -p=500     -e=100     -log=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//qmla.log     -ggr=NVCentreNQubitBath     -run_desc="localdevelopemt"     -git_commit=b221793a00ffd0946995b38e8e3c21d5efcfcb75     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_10/14_49//bayes_factors.csv 

