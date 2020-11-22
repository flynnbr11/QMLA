
cd /home/bf16951/QMD/Launch/Results/Sep_16/22_27/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/22_27/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//system_measurements.p     -ggr=ObjFncElo     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/22_27/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//qmla.log     -ggr=ObjFncElo     -run_desc="localdevelopemt"     -git_commit=b10dfcdce8024a32e3fa0bb0de7fc5a86ec3d29c     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_16/22_27//all_models_bayes_factors.csv 

