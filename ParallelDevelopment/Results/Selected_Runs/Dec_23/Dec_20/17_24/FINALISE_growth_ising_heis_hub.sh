
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24/cumulative.csv -top=3 -qhl=0 -fqhl=0 -data=NVB_rescale_dataset.p -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24/plot_probes.p 	-exp=0 -meas=full_access -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24/true_expec_vals.p -latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Dec_20/17_24/LatexMapping.txt -ggr=hubbard_square_lattice_generalised

