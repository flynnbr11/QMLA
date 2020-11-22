
cd /home/bf16951/QMD/Launch/Results/Sep_10/16_24/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/16_24/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//system_measurements.p     -ggr=ObjFncElo     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=3     -dir=/home/bf16951/QMD/Launch/Results/Sep_10/16_24/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//qmla.log     -ggr=ObjFncElo     -run_desc="localdevelopemt"     -git_commit=16114c70e75e77c67b20db3e6003a6d92cc27636     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_10/16_24//all_models_bayes_factors.csv 

