
cd /home/bf16951/QMD/Launch/Results/Sep_16/10_48/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/10_48/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//system_measurements.p     -ggr=ObjFncAIC     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_16/10_48/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//qmla.log     -ggr=ObjFncAIC     -run_desc="localdevelopemt"     -git_commit=e2ad0c4a5e04629742559e478852e73aa8eafc48     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_16/10_48//all_models_bayes_factors.csv 

