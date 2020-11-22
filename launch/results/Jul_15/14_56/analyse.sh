
cd /home/bf16951/QMD/Launch/Results/Jul_15/14_56/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/14_56/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_15/14_56/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=e6643264274d4439f455868e7e04bded5ff7620c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_15/14_56//bayes_factors.csv 

