
cd /home/bf16951/QMD/Launch/Results/Aug_14/12_22/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_14/12_22/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//system_measurements.p     -ggr=HeisenbergLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=5     -dir=/home/bf16951/QMD/Launch/Results/Aug_14/12_22/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//qmla.log     -ggr=HeisenbergLatticeSet     -run_desc="localdevelopemt"     -git_commit=8a2637d256b0875f927748dead6653862baf6bf2     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_14/12_22//all_models_bayes_factors.csv 

