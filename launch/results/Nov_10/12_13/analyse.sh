
cd /home/bf16951/QMD/Launch/Results/Nov_10/12_13/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/12_13/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/12_13/     -p=1000     -e=200     -log=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=22643968f246b43e4ac7e7acee0de6394a51e46a     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_10/12_13//all_models_bayes_factors.csv 

