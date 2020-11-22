
cd /home/bf16951/QMD/Launch/Results/Jul_13/19_44/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/19_44/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//system_measurements.p     -ggr=NVCentreSimulatedLongDynamicsGenticAlgorithm     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_13/19_44/     -p=100     -e=20     -log=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//qmla.log     -ggr=NVCentreSimulatedLongDynamicsGenticAlgorithm     -run_desc="localdevelopemt"     -git_commit=2bfdc031b8bfa72469d33985006568b7794169c9     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_13/19_44//bayes_factors.csv 

