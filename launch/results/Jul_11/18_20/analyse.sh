
cd /home/bf16951/QMD/Launch/Results/Jul_11/18_20/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_11/18_20/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//system_measurements.p     -ggr=NVCentreNQubitBath     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_11/18_20/     -p=50     -e=10     -log=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//qmla.log     -ggr=NVCentreNQubitBath     -run_desc="localdevelopemt"     -git_commit=3f7e4d900e6eb32848338557a3581abea13e9787     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Jul_11/18_20//bayes_factors.csv 

