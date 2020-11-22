
cd /home/bf16951/QMD/Launch/Results/Jul_29/15_21/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_29/15_21/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_29/15_21/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=95984adc71f7b97be8a329d0f128d12ae8c390d2     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_29/15_21//bayes_factors.csv 

