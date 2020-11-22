
cd /home/bf16951/QMD/Launch/Results/Sep_14/16_57/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_14/16_57/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//system_measurements.p     -ggr=IsingXXZGenetic     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_14/16_57/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//qmla.log     -ggr=IsingXXZGenetic     -run_desc="localdevelopemt"     -git_commit=55743fabb22480a47f477bf20480f7d3568d0040     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_14/16_57//all_models_bayes_factors.csv 

