
cd /home/bf16951/QMD/Launch/Results/Aug_26/10_55/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_26/10_55/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//system_measurements.p     -ggr=IsingGeneticTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_26/10_55/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//qmla.log     -ggr=IsingGeneticTest     -run_desc="localdevelopemt"     -git_commit=d3ef7fa1dd3852d77a35d1917440ff72e0377b5a     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_26/10_55//all_models_bayes_factors.csv 

