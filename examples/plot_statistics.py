"""
Tutorial 02: Plot of statistical data
=========================================
Example script to plot statistical maps, e.g. from correlation analysis
"""
from surface_plot import plot_stats
import numpy as np

outdir = 'data/correlation/test_figures'
clobber = False

# Load data
pval_left = np.loadtxt('https://www.dropbox.com/s/re39ocqymw63gqo/left.pval?dl=1', skiprows=1)
pval_right = np.loadtxt('https://www.dropbox.com/s/a73z66wmori6vtj/right.pval?dl=1', skiprows=1)
tval_left = np.loadtxt('https://www.dropbox.com/s/tv5fei9w37x7lw5/left.tval?dl=1', skiprows=1)
tval_right = np.loadtxt('https://www.dropbox.com/s/as9hfcezvfm9ux8/right.tval?dl=1', skiprows=1)

tval = {'left': tval_left,
        'right': tval_right}

pval = {'left': pval_left,
        'right': pval_right}

# Define degrees of freedom (specific to the example data)
df = 19

# Define statistic threshold
p_threshold = 0.01

