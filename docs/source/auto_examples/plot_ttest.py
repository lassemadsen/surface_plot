"""
Tutorial 03: Plot of mean and statistical data (t-test)
=======================================================
Example script to plot both mean data and statistics, e.g. from paired or unpaied t-test with two groups.
"""
import matplotlib
matplotlib.use('agg')
from surface_plot import plot_mean_stats
import numpy as np

outdir = 'data/paired_ttest/test_figures'
clobber = False

# Load data
pval_left = np.loadtxt('https://www.dropbox.com/s/5g3sopvolf0qnot/left.pval?dl=1', skiprows=1)
pval_right = np.loadtxt('https://www.dropbox.com/s/282xfgqk4wx2pka/right.pval?dl=1', skiprows=1)
tval_left = np.loadtxt('https://www.dropbox.com/s/ku6wdm4nd8690rg/left.tval?dl=1', skiprows=1)
tval_right = np.loadtxt('https://www.dropbox.com/s/oci987u02nwtgxf/right.tval?dl=1', skiprows=1)

mean1_left = np.loadtxt('https://www.dropbox.com/s/htx0q2lm87nnlf1/baseline_left.mean?dl=1', skiprows=1)
mean1_right = np.loadtxt('https://www.dropbox.com/s/mbe1a3v8vtgrlsx/baseline_right.mean?dl=1', skiprows=1)
mean2_left = np.loadtxt('https://www.dropbox.com/s/f71pewt2748dag7/followup_left.mean?dl=1', skiprows=1)
mean2_right = np.loadtxt('https://www.dropbox.com/s/8r8jfkqz9rxjrr3/followup_right.mean?dl=1', skiprows=1)

tval = {'left': tval_left,
        'right': tval_right}

pval = {'left': pval_left,
        'right': pval_right}

mean1 = {'left': mean1_left,
            'right': mean1_right}

mean2 = {'left': mean2_left,
            'right': mean2_right}

# Define statistical threshold
p_threshold = 0.01

# P value
output = f'{outdir}/ttest_pval_p{p_threshold}.pdf'
plot_mean_stats.plot_mean_stats(mean1, mean2, tval, output, pval=pval, vlim_mean=[0, 4], p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Increase', 'Decrease'], clobber=clobber)

output = f'{outdir}/ttest_pval_notitle_p{p_threshold}.pdf'
plot_mean_stats.plot_mean_stats(mean1, mean2, tval, output, pval=pval, vlim_mean=[0, 4], p_threshold=p_threshold, clobber=clobber)

# T value
output = f'{outdir}/ttest_tval_p{p_threshold}.pdf'
plot_mean_stats.plot_mean_stats(mean1, mean2, tval, output, vlim_mean=[0, 4], pval=pval, p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Change'], plot_tvalue=True, clobber=clobber)

output = f'{outdir}/ttest_tval_notitle_p{p_threshold}.pdf'
plot_mean_stats.plot_mean_stats(mean1, mean2, tval, output, pval=pval, vlim_mean=[0, 4], p_threshold=p_threshold, plot_tvalue=True, clobber=clobber)
