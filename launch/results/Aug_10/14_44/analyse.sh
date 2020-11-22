
cd /home/bf16951/QMD/Launch/Results/Aug_10/14_44/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_10/14_44/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//system_measurements.p     -ggr=IsingLatticeSet     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_10/14_44/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//qmla.log     -ggr=IsingLatticeSet     -run_desc="localdevelopemt"     -git_commit=430eba0991c86e3d4848d45d1fcaec54110b0b19     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_10/14_44//bayes_factors.csv 

