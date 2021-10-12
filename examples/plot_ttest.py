"""Example script to plot both mean data and statistics, e.g. from paired or unpaied t-test with two groups.
"""
from surface_plot import plot_mean_stats
import numpy as np

outdir = 'data/paired_ttest/test_figures'
clobber = False

# Load data
pval_left = np.loadtxt('data/paired_ttest/left.pval', skiprows=1)
pval_right = np.loadtxt('data/paired_ttest/right.pval', skiprows=1)
tval_left = np.loadtxt('data/paired_ttest/left.tval', skiprows=1)
tval_right = np.loadtxt('data/paired_ttest/right.tval', skiprows=1)

mean1_left = np.loadtxt('data/paired_ttest/baseline_left.mean', skiprows=1)
mean1_right = np.loadtxt('data/paired_ttest/baseline_right.mean', skiprows=1)
mean2_left = np.loadtxt('data/paired_ttest/followup_left.mean', skiprows=1)
mean2_right = np.loadtxt('data/paired_ttest/followup_right.mean', skiprows=1)

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
output = f'{outdir}/ttest_pval_p{p_threshold}.png'
plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Increase', 'Decrease'], clobber=clobber)

output = f'{outdir}/ttest_pval_notitle_p{p_threshold}.png'
plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, clobber=clobber)

# T value
output = f'{outdir}/ttest_tval_p{p_threshold}.png'
plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Change'], plot_tvalue=True, clobber=clobber)

output = f'{outdir}/ttest_tval_notitle_p{p_threshold}.png'
plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, plot_tvalue=True, clobber=clobber)
