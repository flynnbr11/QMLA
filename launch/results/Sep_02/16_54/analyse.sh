
cd /home/bf16951/QMD/Launch/Results/Sep_02/16_54/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_02/16_54/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//system_measurements.p     -ggr=IsingGeneticTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_02/16_54/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//qmla.log     -ggr=IsingGeneticTest     -run_desc="localdevelopemt"     -git_commit=00620f241f7e39e428bc343a744e0e6c305c7e50     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_02/16_54//all_models_bayes_factors.csv 

