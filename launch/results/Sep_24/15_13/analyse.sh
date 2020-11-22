
cd /home/bf16951/QMD/Launch/Results/Sep_24/15_13/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/15_13/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//system_measurements.p     -ggr=ObjFncResiduals     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_24/15_13/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//qmla.log     -ggr=ObjFncResiduals     -run_desc="localdevelopemt"     -git_commit=e9aae3b7574076fc2233ae21fa756b318d4736c9     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_24/15_13//all_models_bayes_factors.csv 

