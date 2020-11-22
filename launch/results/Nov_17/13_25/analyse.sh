
cd /home/bf16951/QMD/Launch/Results/Nov_17/13_25/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_17/13_25/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_17/13_25/     -p=30     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=98bbb24b82f767c153cfd1edde562e8fe9d5be7c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_17/13_25//all_models_bayes_factors.csv 

