
cd /home/bf16951/QMD/Launch/Results/Sep_15/12_01/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/12_01/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//system_measurements.p     -ggr=ObjFncRank     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/12_01/     -p=1000     -e=200     -log=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//qmla.log     -ggr=ObjFncRank     -run_desc="localdevelopemt"     -git_commit=55743fabb22480a47f477bf20480f7d3568d0040     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_15/12_01//all_models_bayes_factors.csv 

