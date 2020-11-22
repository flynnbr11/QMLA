
cd /home/bf16951/QMD/Launch/Results/Nov_03/18_32/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_03/18_32/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//system_measurements.p     -ggr=HeisenbergGeneticXXZ     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_03/18_32/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//qmla.log     -ggr=HeisenbergGeneticXXZ     -run_desc="localdevelopemt"     -git_commit=64037f7c53d241918d5d5b86556ac64a7e63fa14     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_03/18_32//all_models_bayes_factors.csv 

