
cd /home/bf16951/QMD/Launch/Results/Aug_15/17_29/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_15/17_29/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=5     -dir=/home/bf16951/QMD/Launch/Results/Aug_15/17_29/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=802a2335f2afd56725435cbec41998e18be8b521     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_15/17_29//all_models_bayes_factors.csv 

