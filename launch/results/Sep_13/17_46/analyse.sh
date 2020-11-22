
cd /home/bf16951/QMD/Launch/Results/Sep_13/17_46/
python3 ../../../../scripts/analyse_qmla.py     -dir=/home/bf16951/QMD/Launch/Results/Sep_13/17_46/     --bayes_csv=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//all_models_bayes_factors.csv     -log=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//qmla.log     -top=5     -qhl=0     -fqhl=0     -runinfo=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//run_info.p     -sysmeas=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//system_measurements.p     -ggr=ObjFncResiduals     -plotprobes=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//plot_probes.p     -latex=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//latex_mapping.txt     -gs=1

python3 ../../../../scripts/generate_results_pdf.py     -t=1     -dir=/home/bf16951/QMD/Launch/Results/Sep_13/17_46/     -p=10     -e=2     -log=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//qmla.log     -ggr=ObjFncResiduals     -run_desc="localdevelopemt"     -git_commit=befc07caae63e469e41a26653bf6d21db02f8698     -qhl=0     -mqhl=0     -cb=/home/bf16951/QMD/Launch/Results/Sep_13/17_46//all_models_bayes_factors.csv 

