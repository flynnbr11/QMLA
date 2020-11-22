
cd /home/bf16951/QMD/Launch/Results/Oct_01/15_52/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Oct_01/15_52/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//system_measurements.p     -ggr=NVPrelearnedTest     -plotprobes=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Oct_01/15_52/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//qmla.log     -ggr=NVPrelearnedTest     -run_desc="localdevelopemt"     -git_commit=18251416c5498e025bcdfbf6c4cbce1e796f286e     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Oct_01/15_52//all_models_bayes_factors.csv 

