
#!/bin/bash 
cd /home/bf16951/Dropbox/QML_share_stateofart/QMD/Libraries/QML_lib

python3 AnalyseMultipleQMD.py -dir=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01 --bayes_csv=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01/cumulative.csv -top=3 -qhl=0 -fqhl=0 -data=NVB_rescale_dataset.p -plot_probes=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01/plot_probes.p 	-exp=0 -meas=full_access -params=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01/true_params.p -true_expec=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01/true_expec_vals.p -latex=/home/bf16951/Dropbox/QML_share_stateofart/QMD/ParallelDevelopment/Results/Dec_20/14_01/LatexMapping.txt -ggr=hubbard_square_lattice_generalised

