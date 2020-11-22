
cd /home/bf16951/QMD/Launch/Results/Sep_10/15_42/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/15_42/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//system_measurements.p     -ggr=ObjFncElo     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=3     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/15_42/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//qmla.log     -ggr=ObjFncElo     -run_desc="localdevelopemt"     -git_commit=16114c70e75e77c67b20db3e6003a6d92cc27636     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_10/15_42//all_models_bayes_factors.csv 

