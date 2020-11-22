
cd /home/bf16951/QMD/Launch/Results/Nov_21/12_32/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_21/12_32/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_21/12_32/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=ff5ddaae8ef3b359474f161b8b0f7e774a0d79cc     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_21/12_32//all_models_bayes_factors.csv 

