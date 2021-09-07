import logging
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

import plot_stats
import plot_mean_stats

logger = logging.getLogger('test')

def main():

    correlation_plot()

    ttest_plot()

def correlation_plot():
    outdir = '../data/correlation/test_figures'
    clobber = False

    pval_left = np.loadtxt('../data/correlation/left.pval', skiprows=1)
    pval_right = np.loadtxt('../data/correlation/right.pval', skiprows=1)
    tval_left = np.loadtxt('../data/correlation/left.tval', skiprows=1)
    tval_right = np.loadtxt('../data/correlation/right.tval', skiprows=1)

    tval = {'left': tval_left,
            'right': tval_right}

    pval = {'left': pval_left,
            'right': pval_right}

    for p_threshold in [0.01, 0.001]:

        # T value test plots
        for cbar_loc in ['left', 'bottom', None]:
            output = '{}/tval_cbar_{}_p{}.png'.format(outdir, cbar_loc, p_threshold)
            plot_stats.plot_stats(pval, tval, output, plot_tvalue=True, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

        # P value test plots
        for cbar_loc in ['left', 'bottom', None]:
            output = '{}/pval_cbar_{}_p{}.png'.format(outdir, cbar_loc, p_threshold)
            plot_stats.plot_stats(pval, tval, output, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

            output = '{}/pval_cbar_{}_p{}_notitle.png'.format(outdir, cbar_loc, p_threshold)
            plot_stats.plot_stats(pval, tval, output, p_threshold=p_threshold, cbar_loc=cbar_loc, titles=None, clobber=clobber)

def ttest_plot():
    outdir = '../data/paired_ttest/test_figures'
    clobber = False

    pval_left = np.loadtxt('../data/paired_ttest/left.pval', skiprows=1)
    pval_right = np.loadtxt('../data/paired_ttest/right.pval', skiprows=1)
    tval_left = np.loadtxt('../data/paired_ttest/left.tval', skiprows=1)
    tval_right = np.loadtxt('../data/paired_ttest/right.tval', skiprows=1)

    mean1_left = np.loadtxt('../data/paired_ttest/baseline_left.mean', skiprows=1)
    mean1_right = np.loadtxt('../data/paired_ttest/baseline_right.mean', skiprows=1)
    mean2_left = np.loadtxt('../data/paired_ttest/baseline_left.mean', skiprows=1)
    mean2_right = np.loadtxt('../data/paired_ttest/followup_right.mean', skiprows=1)

    tval = {'left': tval_left,
            'right': tval_right}

    pval = {'left': pval_left,
            'right': pval_right}

    mean1 = {'left': mean1_left,
             'right': mean1_right}

    mean2 = {'left': mean2_left,
             'right': mean2_right}

    for p_threshold in [0.01, 0.001]:

        # P value
        output = '{}/ttest_pval_p{}.png'.format(outdir, p_threshold)
        plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Increase', 'Decrease'], clobber=clobber)

        output = '{}/ttest_pval_notitle_p{}.png'.format(outdir, p_threshold)
        plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, clobber=clobber)

        # T value
        output = '{}/ttest_tval_p{}.png'.format(outdir, p_threshold)
        plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, cb_mean_title='Mean SUVR', mean_titles=['Baseline', 'Follow-up'], stats_titles=['Change'], plot_tvalue=True, clobber=clobber)

        output = '{}/ttest_tval_notitle_p{}.png'.format(outdir, p_threshold)
        plot_mean_stats.plot_mean_stats(mean1, mean2, pval, tval, output, vlim_mean=[0, 4], p_threshold=p_threshold, plot_tvalue=True, clobber=clobber)




if __name__ == "__main__":
    main()
