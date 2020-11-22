
cd /home/bf16951/QMD/Launch/Results/Aug_29/18_00/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_29/18_00/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//system_measurements.p     -ggr=Demonstration     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_29/18_00/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//qmla.log     -ggr=Demonstration     -run_desc="localdevelopemt"     -git_commit=704d93e3f4b4ae149e061dbfd4369e441a28ff0b     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_29/18_00//all_models_bayes_factors.csv 

