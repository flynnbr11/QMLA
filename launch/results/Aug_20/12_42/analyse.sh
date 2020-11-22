
cd /home/bf16951/QMD/Launch/Results/Aug_20/12_42/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_20/12_42/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//system_measurements.p     -ggr=HeisenbergGenetic     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_20/12_42/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//qmla.log     -ggr=HeisenbergGenetic     -run_desc="localdevelopemt"     -git_commit=db6a3cb9b4085dfab6f6f56c08331503a4e6f295     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_20/12_42//all_models_bayes_factors.csv 

