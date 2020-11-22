
cd /home/bf16951/QMD/Launch/Results/Aug_16/18_47/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_16/18_47/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=5     -dir=/home/bf16951/QMD/Launch/Results/Aug_16/18_47/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=ec747cf047bb5e48804cd80fc7c9680ef04fa3d1     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Aug_16/18_47//all_models_bayes_factors.csv 

