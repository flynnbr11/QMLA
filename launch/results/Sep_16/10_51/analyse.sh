
cd /home/bf16951/QMD/Launch/Results/Sep_16/10_51/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/10_51/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//system_measurements.p     -ggr=ObjFncAIC     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/10_51/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//qmla.log     -ggr=ObjFncAIC     -run_desc="localdevelopemt"     -git_commit=2e24b7106eea0730ea050877d4f5e7dc411ead89     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_16/10_51//all_models_bayes_factors.csv 

