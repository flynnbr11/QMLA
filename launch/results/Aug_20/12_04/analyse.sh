
cd /home/bf16951/QMD/Launch/Results/Aug_20/12_04/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_20/12_04/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//system_measurements.p     -ggr=HeisenbergGenetic     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_20/12_04/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//qmla.log     -ggr=HeisenbergGenetic     -run_desc="localdevelopemt"     -git_commit=db6a3cb9b4085dfab6f6f56c08331503a4e6f295     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_20/12_04//all_models_bayes_factors.csv 

