
cd /home/bf16951/QMD/Launch/Results/Jul_13/12_12/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/12_12/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/12_12/     -p=1000     -e=200     -log=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=5f6e71ec2ae9a5f2ac992382f67e50a3602c1ebe     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_13/12_12//bayes_factors.csv 

