
cd /home/bf16951/QMD/Launch/Results/Sep_24/17_59/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/17_59/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/17_59/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=83d5c9688894599de5e3adf60af8086a5ff6895a     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_24/17_59//all_models_bayes_factors.csv 

