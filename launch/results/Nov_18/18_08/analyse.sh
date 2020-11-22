
cd /home/bf16951/QMD/Launch/Results/Nov_18/18_08/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_18/18_08/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_18/18_08/     -p=20     -e=10     -log=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=3f3e858b2723ac11034f5f92824aecf20d98bd88     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_18/18_08//all_models_bayes_factors.csv 

