
cd /home/bf16951/QMD/Launch/Results/Aug_10/11_51/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_10/11_51/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_10/11_51/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3bf5ad7d1f61372008452d37df912e2f964fbdac     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_10/11_51//bayes_factors.csv 

