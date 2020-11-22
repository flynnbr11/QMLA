
cd /home/bf16951/QMD/Launch/Results/Aug_19/15_20/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/15_20/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/15_20/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=28eb617dba7cfbdad5230e67e74f64d1c1a88923     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_19/15_20//all_models_bayes_factors.csv 

