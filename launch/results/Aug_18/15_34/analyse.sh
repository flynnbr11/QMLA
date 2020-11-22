
cd /home/bf16951/QMD/Launch/Results/Aug_18/15_34/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/15_34/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/15_34/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=2fb4b09d75e477fe8460ed59b34b7fe95783a39f     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_18/15_34//all_models_bayes_factors.csv 

