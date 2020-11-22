
cd /home/bf16951/QMD/Launch/Results/Sep_16/18_41/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/18_41/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//system_measurements.p     -ggr=ObjFncElo     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/18_41/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//qmla.log     -ggr=ObjFncElo     -run_desc="localdevelopemt"     -git_commit=54ee82d2c1d6ae759f04c8976c321f51e537a3ef     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_16/18_41//all_models_bayes_factors.csv 

