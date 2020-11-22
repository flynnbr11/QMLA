
cd /home/bf16951/QMD/Launch/Results/Aug_19/22_05/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/22_05/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=2     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/22_05/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=a3fc337f96dfef77b801471142b44a1c5c6528c1     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_19/22_05//all_models_bayes_factors.csv 

