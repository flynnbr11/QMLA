
cd /home/bf16951/QMD/Launch/Results/Aug_18/16_27/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/16_27/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/16_27/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=bfd187ce0655a047d06637cc7d6af9704e7a2da5     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_18/16_27//all_models_bayes_factors.csv 

