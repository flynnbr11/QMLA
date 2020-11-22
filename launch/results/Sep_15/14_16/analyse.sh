
cd /home/bf16951/QMD/Launch/Results/Sep_15/14_16/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/14_16/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//system_measurements.p     -ggr=ObjFncRank     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/14_16/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//qmla.log     -ggr=ObjFncRank     -run_desc="localdevelopemt"     -git_commit=722e0f0b1c099d340f9c9e3e814a3563d39a7469     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_15/14_16//all_models_bayes_factors.csv 

