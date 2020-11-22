
cd /home/bf16951/QMD/Launch/Results/Aug_18/15_46/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/15_46/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_18/15_46/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=21d316813f9282d5874f83724efa5bb3675d23f9     -qhl=0     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Aug_18/15_46//all_models_bayes_factors.csv 

