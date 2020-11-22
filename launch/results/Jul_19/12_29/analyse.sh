
cd /home/bf16951/QMD/Launch/Results/Jul_19/12_29/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Jul_19/12_29/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//system_measurements.p     -ggr=IsingGeneticTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Jul_19/12_29/     -p=2     -e=2     -log=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//qmla.log     -ggr=IsingGeneticTest     -run_desc="localdevelopemt"     -git_commit=3105de52b14e39a0920f79e0f3e7ecb7f0d879e5     -qhl=1     -mqhl=1     -cb=/home/bf16951/QMD/Launch/Results/Jul_19/12_29//bayes_factors.csv 

