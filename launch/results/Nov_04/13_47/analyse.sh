
cd /home/bf16951/QMD/Launch/Results/Nov_04/13_47/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_04/13_47/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_04/13_47/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=16e34e2fc568fa78bb44aa04db906db29e208d2f     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_04/13_47//all_models_bayes_factors.csv 

