
cd /home/bf16951/QMD/Launch/Results/Sep_17/21_42/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_17/21_42/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_17/21_42/     -p=2000     -e=500     -log=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=fd858d6e314e25959995ea626654a8b6d951f11c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_17/21_42//all_models_bayes_factors.csv 

