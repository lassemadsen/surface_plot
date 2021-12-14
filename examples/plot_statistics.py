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

# --- Plot p-values ---
for cbar_loc in ['left', 'bottom', None]:
    # Plot two-tailed pval (i.e. without knowing wheter its is right- or left tailed)
    output = f'{outdir}/pval_twotailed_cbar_{cbar_loc}_p{p_threshold}.pdf'
    plot_stats.plot_pval(pval, output, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

    # Plot seperate one-tailed pval (i.e. one plot with right tail (positive) and one plot with left tail (negative)) 
    output = f'{outdir}/pval_onetailed_cbar_{cbar_loc}_p{p_threshold}.pdf'
    plot_stats.plot_pval(pval, output, tval=tval, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

# --- Plot t-values ---
t_threshold = 2.5
p_threshold = 0.01
cbar_loc = 'left'

# Plot t-values with automatic limits
output = f'{outdir}/tval_autolim_cbar_{cbar_loc}_t{t_threshold}.pdf'
plot_stats.plot_tval(tval, output, t_threshold=t_threshold, cbar_loc=cbar_loc, clobber=clobber)

# Plot t-values with predetermined limits
t_lim = [-3, 3]
output = f'{outdir}/tval_withlim_cbar_{cbar_loc}_t{t_threshold}.pdf'
plot_stats.plot_tval(tval, output, t_lim=t_lim, t_threshold=0, cbar_loc=cbar_loc, clobber=clobber)

# Plot t-values with p_threshold and calculated p-values
output = f'{outdir}/tval_withpval_cbar_{cbar_loc}_p{p_threshold}.pdf'
plot_stats.plot_tval(tval, output, p_threshold=p_threshold, pval=pval, cbar_loc=cbar_loc, clobber=clobber)

# Plot t-values with p_threshold and df, two-tailed
output = f'{outdir}/tval_withdf_twotailed_cbar_{cbar_loc}_p{p_threshold}.pdf'
plot_stats.plot_tval(tval, output, p_threshold=p_threshold, df=df, cbar_loc=cbar_loc, clobber=clobber)

# Plot t-values with p_threshold and df, one-tailed
output = f'{outdir}/tval_withdf_onetailed_cbar_{cbar_loc}_p{p_threshold}.pdf'
plot_stats.plot_tval(tval, output, p_threshold=p_threshold, df=df, two_tailed=False, cbar_loc=cbar_loc, clobber=clobber)