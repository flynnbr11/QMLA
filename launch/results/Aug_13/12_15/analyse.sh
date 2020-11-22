
cd /home/bf16951/QMD/Launch/Results/Aug_13/12_15/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_13/12_15/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_13/12_15/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=e6c3f613be3bb8be0ddd823204646c0454457bbf     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_13/12_15//all_models_bayes_factors.csv 

