
cd /home/bf16951/QMD/Launch/Results/Jul_13/15_25/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/15_25/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/15_25/     -p=1000     -e=200     -log=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=fd4cc087072108fee7abe554817e1205373b24d3     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_13/15_25//bayes_factors.csv 

