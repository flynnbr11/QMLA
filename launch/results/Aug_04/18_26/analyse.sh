
cd /home/bf16951/QMD/Launch/Results/Aug_04/18_26/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_04/18_26/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_04/18_26/     -p=100     -e=25     -log=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=3f14f4dce7816a96c94939fdbc9910aa9c005c39     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_04/18_26//bayes_factors.csv 

