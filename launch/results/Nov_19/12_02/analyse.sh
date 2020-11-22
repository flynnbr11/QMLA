
cd /home/bf16951/QMD/Launch/Results/Nov_19/12_02/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_19/12_02/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_19/12_02/     -p=20     -e=10     -log=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=0b1c924ab21492ab01edcfecf796b7dc051213d4     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_19/12_02//all_models_bayes_factors.csv 

