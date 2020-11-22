
cd /home/bf16951/QMD/Launch/Results/Sep_22/14_56/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_22/14_56/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_22/14_56/     -p=2500     -e=500     -log=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=1e3c49f6f330f1e956cf23dcfae8ea276b3c55d4     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_22/14_56//all_models_bayes_factors.csv 

