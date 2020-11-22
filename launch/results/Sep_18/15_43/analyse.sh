
cd /home/bf16951/QMD/Launch/Results/Sep_18/15_43/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_18/15_43/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_18/15_43/     -p=300     -e=750     -log=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=402fca548c347dac14f7354679ddc7a943a52a55     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_18/15_43//all_models_bayes_factors.csv 

