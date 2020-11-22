
cd /home/bf16951/QMD/Launch/Results/Jul_30/17_28/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/17_28/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_30/17_28/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=50b514ca0168473ea443a94182415bee0e093a28     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_30/17_28//bayes_factors.csv 

