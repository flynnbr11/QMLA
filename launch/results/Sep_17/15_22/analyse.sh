
cd /home/bf16951/QMD/Launch/Results/Sep_17/15_22/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_17/15_22/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_17/15_22/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=354840622ca6bd849dd997c287cc0b0e5ae1619d     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_17/15_22//all_models_bayes_factors.csv 

