
cd /home/bf16951/QMD/Launch/Results/Jul_30/16_09/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/16_09/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/16_09/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=9b2cfdd30a7337d4eadde3c121f64c22c6ac59f7     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_30/16_09//bayes_factors.csv 

