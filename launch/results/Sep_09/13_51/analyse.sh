
cd /home/bf16951/QMD/Launch/Results/Sep_09/13_51/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_09/13_51/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_09/13_51/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=13e592b0d67713d9bed29c87c27f6118cfcc6467     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_09/13_51//all_models_bayes_factors.csv 

