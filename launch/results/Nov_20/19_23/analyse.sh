
cd /home/bf16951/QMD/Launch/Results/Nov_20/19_23/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_20/19_23/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_20/19_23/     -p=2000     -e=500     -log=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=5ebbaa47ae5e82e4f2b4c715fd73e2369e18806c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_20/19_23//all_models_bayes_factors.csv 

