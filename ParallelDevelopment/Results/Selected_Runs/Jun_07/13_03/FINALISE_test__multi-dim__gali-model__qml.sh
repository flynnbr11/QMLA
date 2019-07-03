
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/cumulative.csv -top=3 -qhl=0 -fqhl=0 -plot_probes=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/plot_probes.p 	-exp=0 -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/true_expec_vals.p -latex=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/LatexMapping.txt -ggr=NV_centre_spin_large_bath

python3 CombineAnalysisPlots.py     -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03     -p=2000 -e=500 -bt=500 -t=25     -nprobes=40     -pnoise=0.0000001     -special_probe=random     -ggr=NV_centre_spin_large_bath     -run_desc=test__multi-dim__gali-model__qml     -git_commit=9652cd0956f8be2ef5621145b1bf14ec4f2ffacf     -ra=0.98     -rt=0.5     -pgh=1.0     -qhl=0     -mqhl=0     -cb=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Jun_07/13_03/cumulative.csv     -exp=0

