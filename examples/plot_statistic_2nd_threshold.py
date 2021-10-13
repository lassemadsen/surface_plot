"""
Tutorial 04: Plot of statistical maps with two thresholds
=========================================================
Example script to plot statistical maps with two different statistical thresholds.
The first threshold is plottet as normal, the second threshold is outlined by a white line.

The second_level data is defined by a mask containing e.g. surviving clusters (1's)

Note: the cortical surface template is needed to segment the clusters. 
"""
from surface_plot import plot_stats
import numpy as np
from brainspace.mesh.mesh_io import read_surface

outdir = 'data/second_level/test_figures'
clobber = True
mask = {'left': [], 'right': []}

surf = {'left': read_surface('/Users/au483096/data/atlas/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii'),
        'right': read_surface('/Users/au483096/data/atlas/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii')}

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

output = f'{outdir}/second_level.png'

plot_stats.plot_tval(tval, output, mask=mask, t_lim=t_lim, second_threshold_mask=second_level, surf=surf, expand_edge=True, clobber=clobber)
