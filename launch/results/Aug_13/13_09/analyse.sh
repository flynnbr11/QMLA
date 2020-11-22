
cd /home/bf16951/QMD/Launch/Results/Aug_13/13_09/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_13/13_09/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_13/13_09/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=e6c3f613be3bb8be0ddd823204646c0454457bbf     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_13/13_09//all_models_bayes_factors.csv 

