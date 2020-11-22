
cd /home/bf16951/QMD/Launch/Results/Nov_10/12_00/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/12_00/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Nov_10/12_00/     -p=20     -e=5     -log=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=22643968f246b43e4ac7e7acee0de6394a51e46a     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Nov_10/12_00//all_models_bayes_factors.csv 

