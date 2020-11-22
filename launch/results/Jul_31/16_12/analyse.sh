
cd /home/bf16951/QMD/Launch/Results/Jul_31/16_12/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_31/16_12/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_31/16_12/     -p=100     -e=25     -log=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=54e35f95d3e40092452583350f14c0903e828a89     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_31/16_12//bayes_factors.csv 

