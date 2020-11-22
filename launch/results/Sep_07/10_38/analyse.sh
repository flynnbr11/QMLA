
cd /home/bf16951/QMD/Launch/Results/Sep_07/10_38/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_07/10_38/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//system_measurements.p     -ggr=IsingGeneticSingleLayer     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_07/10_38/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//qmla.log     -ggr=IsingGeneticSingleLayer     -run_desc="localdevelopemt"     -git_commit=fec6a81d0c5b0db8f746f43902b2904af6788e76     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_07/10_38//all_models_bayes_factors.csv 

