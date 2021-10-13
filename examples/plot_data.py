"""
Tutorial 01: Simple plot of cortical surface data
=================================================
Example script to plot simple data to cortical surface
"""
from surface_plot import plot_surface
import numpy as np

outdir = 'data/simple_plot/test_figures'
clobber = False

# Load data
data_left = np.loadtxt('https://www.dropbox.com/s/lezvpw14iw13d9r/data_left.mean?dl=1', skiprows=1)
data_right = np.loadtxt('https://www.dropbox.com/s/lezvpw14iw13d9r/data_right.mean?dl=1', skiprows=1)

data = {'left': data_left,
        'right': data_right}

# Plot mean data - automatic limits
output = f'{outdir}/simple_autolim.png'
plot_surface.plot_surface(data, output, clobber=clobber)

# Plot mean data - predetermined limits
output = f'{outdir}/simple_withlim.png'
plot_surface.plot_surface(data, output, vlim=[0, 2.5], cbar_loc='bottom', clobber=clobber)

# Plot mean data - predetermined limits with title
output = f'{outdir}/simple_withlim_title.png'
plot_surface.plot_surface(data, output, vlim=[0.5, 3], cbar_loc='bottom', title='PiB uptake', cbar_title='Mean SUVR', clobber=clobber)
