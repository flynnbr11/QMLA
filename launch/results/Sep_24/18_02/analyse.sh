
cd /home/bf16951/QMD/Launch/Results/Sep_24/18_02/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/18_02/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/18_02/     -p=1500     -e=250     -log=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=4a37852c0ab358e152dd3de458138e89ea415a0d     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_24/18_02//all_models_bayes_factors.csv 

