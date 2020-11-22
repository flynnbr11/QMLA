
cd /home/bf16951/QMD/Launch/Results/Nov_22/12_22/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_22/12_22/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_22/12_22/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=feb51247545bb4626bae88ace11e6d8f3416ee8c     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_22/12_22//all_models_bayes_factors.csv 

