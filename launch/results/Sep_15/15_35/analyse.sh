
cd /home/bf16951/QMD/Launch/Results/Sep_15/15_35/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/15_35/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//qmla.log     -top=5     -qhl=1     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//system_measurements.p     -ggr=ObjFncAIC     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_15/15_35/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//qmla.log     -ggr=ObjFncAIC     -run_desc="localdevelopemt"     -git_commit=ab63b567ab9dbe30829fc347c68d35ac44da02d3     -qhl=1     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_15/15_35//all_models_bayes_factors.csv 

