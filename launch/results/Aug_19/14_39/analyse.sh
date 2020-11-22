
cd /home/bf16951/QMD/Launch/Results/Aug_19/14_39/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/14_39/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//system_measurements.p     -ggr=FermiHubbardLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=5     -dir=/home/bf16951/QMD/Launch/Results/Aug_19/14_39/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//qmla.log     -ggr=FermiHubbardLatticeSet     -run_desc="localdevelopemt"     -git_commit=7dc8138fe0d15cdbabbc7176b1a913eab882a21e     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_19/14_39//all_models_bayes_factors.csv 

