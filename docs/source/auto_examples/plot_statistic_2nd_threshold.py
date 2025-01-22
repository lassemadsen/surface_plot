"""
Tutorial 04: Plot of statistical maps with two thresholds
=========================================================
Example script to plot statistical maps with two different statistical thresholds.
The first threshold is plottet as normal, the second threshold is outlined by a white line.

The second_level data is defined by a mask containing e.g. surviving clusters (1's)

Note: the cortical surface template is needed to segment the clusters. 
"""
import matplotlib
matplotlib.use('agg')
from surface_plot import plot_stats
import numpy as np

outdir = 'data/second_level/test_figures'
clobber = True
mask = {'left': [], 'right': []}

# Load data
tval_left = np.loadtxt('https://www.dropbox.com/s/5ek63zf5l2iwd8f/tval_left.csv?dl=1')
tval_right = np.loadtxt('https://www.dropbox.com/s/def320uevw5ivsb/tval_right.csv?dl=1')
second_level_left = np.loadtxt('https://www.dropbox.com/s/y93z65g2jx0zs2x/second_level_left.csv?dl=1')
second_level_right = np.loadtxt('https://www.dropbox.com/s/wgp0j0gp35oe41g/second_level_right.csv?dl=1')

tval = {'left': tval_left,
        'right': tval_right}

second_level = {'left': second_level_left,
                'right': second_level_right}

t_lim = [-5, 5]

mask['left'] = ~np.isnan(tval['left'])
mask['right'] = ~np.isnan(tval['right'])

output = f'{outdir}/second_level.pdf'

plot_stats.plot_tval(tval, output, mask=mask, t_lim=t_lim, second_threshold_mask=second_level, expand_edge=True, clobber=clobber)
