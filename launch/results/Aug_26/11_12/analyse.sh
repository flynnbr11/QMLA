
cd /home/bf16951/QMD/Launch/Results/Aug_26/11_12/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_26/11_12/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//system_measurements.p     -ggr=IsingGeneticTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_26/11_12/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//qmla.log     -ggr=IsingGeneticTest     -run_desc="localdevelopemt"     -git_commit=d3ef7fa1dd3852d77a35d1917440ff72e0377b5a     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_26/11_12//all_models_bayes_factors.csv 

