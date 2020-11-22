
cd /home/bf16951/QMD/Launch/Results/Nov_10/18_06/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/18_06/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/18_06/     -p=1000     -e=200     -log=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=dc4cfe736e6af795f9cc8156aa49415c9789f48a     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_10/18_06//all_models_bayes_factors.csv 

