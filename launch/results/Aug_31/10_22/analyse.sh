
cd /home/bf16951/QMD/Launch/Results/Aug_31/10_22/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_31/10_22/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_31/10_22/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=4b83ee7c8119e5472a48750c1e3ccf8a40c71485     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_31/10_22//all_models_bayes_factors.csv 

