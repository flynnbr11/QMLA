
cd /home/bf16951/QMD/Launch/Results/Nov_13/21_54/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_13/21_54/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_13/21_54/     -p=100     -e=20     -log=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=244f7ce0ab791f419a2dec9ab14bbcff9893899d     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_13/21_54//all_models_bayes_factors.csv 

