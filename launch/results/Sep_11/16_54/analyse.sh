
cd /home/bf16951/QMD/Launch/Results/Sep_11/16_54/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_11/16_54/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//system_measurements.p     -ggr=ObjFncRank     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=2     -dir=/home/bf16951/QMD/Launch/Results/Sep_11/16_54/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//qmla.log     -ggr=ObjFncRank     -run_desc="localdevelopemt"     -git_commit=a6ad8c13f97776f14b6a0de1395ec7b4487d9cdd     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_11/16_54//all_models_bayes_factors.csv 

