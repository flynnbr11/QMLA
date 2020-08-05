import numpy as np
import argparse
from matplotlib.lines import Line2D
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pandas
import warnings

plt.switch_backend('agg')
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..')
    )
)
import qmla
import qmla.analysis

test = False
if test:
    run_path = '/home/bf16951/QMD/Launch/Results/Aug_05/15_23'
    focus_on_instance = '001'
    branches_to_draw = [1]
else:
    run_path = '/panfs/panasas01/phys/bf16951/QMD/Launch/Results/Aug_05/15_59/'
    focus_on_instance = '013'
    branches_to_draw = [1, 15, 30, 45]

save_to_file = os.path.join(run_path, 'analysis_figure_3.pdf')

qmla.analysis.nv_centre_experimental_paper_fig_3(
    run_path,
    focus_on_instance,
    branches_to_draw=branches_to_draw,
    save_to_file=save_to_file, 
)