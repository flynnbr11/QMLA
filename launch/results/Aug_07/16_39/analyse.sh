
cd /home/bf16951/QMD/Launch/Results/Aug_07/16_39/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Aug_07/16_39/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//system_measurements.p     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -plotprobes=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Aug_07/16_39/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//qmla.log     -ggr=NVCentreGenticAlgorithmPrelearnedParameters     -run_desc="localdevelopemt"     -git_commit=4ec1edc41d6691ea63865836dae056101d0a0df2     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Aug_07/16_39//bayes_factors.csv 

