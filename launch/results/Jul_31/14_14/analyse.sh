
cd /home/bf16951/QMD/Launch/Results/Jul_31/14_14/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_31/14_14/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_31/14_14/     -p=1000     -e=250     -log=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=77ffd7576749ca0259ee0426d4137bbad4a18223     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_31/14_14//bayes_factors.csv 

