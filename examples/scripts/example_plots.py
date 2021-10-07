from surface_plot import plot_stats, plot_mean_stats, plot_surface
import numpy as np
import pandas as pd
from brainstat.mesh.interpolate import read_surface_gz

def main():

    data_plot()

    correlation_plot()
 
    ttest_plot()

    second_level()

def data_plot():
    outdir = '../data/simple_plot/test_figures'

    data_left = np.loadtxt('../data/simple_plot/data_left.mean', skiprows=1)
    data_right = np.loadtxt('../data/simple_plot/data_right.mean', skiprows=1)

    data = {'left': data_left,
            'right': data_right}

    # Plot mean data - automatic limits
    output = f'{outdir}/simple_autolim.png'
    plot_surface.plot_surface(data, output)

    # Plot mean data - predetermined limits
    output = f'{outdir}/simple_withlim.png'
    plot_surface.plot_surface(data, output, vlim=[0, 2.5], cbar_loc='bottom')

    # Plot mean data - predetermined limits with title
    output = f'{outdir}/simple_withlim_title.png'
    plot_surface.plot_surface(data, output, vlim=[0.5, 3], cbar_loc='bottom', title='PiB uptake', cbar_title='Mean SUVR')

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

    df = 19
    p_threshold = 0.01

    # --- Plot p-values ---
    for cbar_loc in ['left', 'bottom', None]:
        # Plot two-tailed pval (i.e. without knowing wheter its is right- or left tailed)
        output = f'{outdir}/pval_twotailed_cbar_{cbar_loc}_p{p_threshold}.png'
        plot_stats.plot_pval(pval, output, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

        # Plot seperate one-tailed pval (i.e. one plot with right tail (positive) and one plot with left tail (negative)) 
        output = f'{outdir}/pval_onetailed_cbar_{cbar_loc}_p{p_threshold}.png'
        plot_stats.plot_pval(pval, output, tval=tval, p_threshold=p_threshold, cbar_loc=cbar_loc, clobber=clobber)

    # --- Plot t-values ---
    t_threshold = 2.5
    p_threshold = 0.01
    cbar_loc = 'left'

    # Plot t-values with automatic limits
    output = f'{outdir}/tval_autolim_cbar_{cbar_loc}_t{t_threshold}.png'
    plot_stats.plot_tval(tval, output, t_threshold=t_threshold, cbar_loc=cbar_loc, clobber=clobber)

    # Plot t-values with predetermined limits
    t_lim = [-3, 3]
    output = f'{outdir}/tval_withlim_cbar_{cbar_loc}_t{t_threshold}.png'
    plot_stats.plot_tval(tval, output, t_lim=t_lim, t_threshold=0, cbar_loc=cbar_loc, clobber=clobber)

    # Plot t-values with p_threshold and calculated p-values
    output = f'{outdir}/tval_withpval_cbar_{cbar_loc}_p{p_threshold}.png'
    plot_stats.plot_tval(tval, output, p_threshold=p_threshold, pval=pval, cbar_loc=cbar_loc, clobber=clobber)

    # Plot t-values with p_threshold and df, two-tailed
    output = f'{outdir}/tval_withdf_twotailed_cbar_{cbar_loc}_p{p_threshold}.png'
    plot_stats.plot_tval(tval, output, p_threshold=p_threshold, df=df, cbar_loc=cbar_loc, clobber=clobber)

    # Plot t-values with p_threshold and df, one-tailed
    output = f'{outdir}/tval_withdf_onetailed_cbar_{cbar_loc}_p{p_threshold}.png'
    plot_stats.plot_tval(tval, output, p_threshold=p_threshold, df=df, two_tailed=False, cbar_loc=cbar_loc, clobber=clobber)

def ttest_plot():
    outdir = '../data/paired_ttest/test_figures'
    clobber = False

    pval_left = np.loadtxt('../data/paired_ttest/left.pval', skiprows=1)
    pval_right = np.loadtxt('../data/paired_ttest/right.pval', skiprows=1)
    tval_left = np.loadtxt('../data/paired_ttest/left.tval', skiprows=1)
    tval_right = np.loadtxt('../data/paired_ttest/right.tval', skiprows=1)

    mean1_left = np.loadtxt('../data/paired_ttest/baseline_left.mean', skiprows=1)
    mean1_right = np.loadtxt('../data/paired_ttest/baseline_right.mean', skiprows=1)
    mean2_left = np.loadtxt('../data/paired_ttest/followup_left.mean', skiprows=1)
    mean2_right = np.loadtxt('../data/paired_ttest/followup_right.mean', skiprows=1)

    tval = {'left': tval_left,
            'right': tval_right}

    pval = {'left': pval_left,
            'right': pval_right}

    mean1 = {'left': mean1_left,
             'right': mean1_right}

    mean2 = {'left': mean2_left,
             'right': mean2_right}

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

def second_level():
    outdir = '../data/second_level/test_figures'
    clobber = True
    mask = {'left': [], 'right': []}

    surf = {'left': read_surface_gz('/Users/au483096/data/atlas/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii'),
            'right': read_surface_gz('/Users/au483096/data/atlas/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii')}

    tval_left = np.loadtxt('../data/second_level/tval_left.csv')
    tval_right = np.loadtxt('../data/second_level/tval_right.csv')
    second_level_left = np.loadtxt('../data/second_level/second_level_left.csv')
    second_level_right = np.loadtxt('../data/second_level/second_level_right.csv')

    tval = {'left': tval_left,
            'right': tval_right}

    second_level = {'left': second_level_left,
                    'right': second_level_right}

    df = 19
    p_threshold = 0.01 
    t_lim = [-5, 5]

    mask['left'] = ~np.isnan(tval['left'])
    mask['right'] = ~np.isnan(tval['right'])

    output = f'{outdir}/second_level.png'

    plot_stats.plot_tval(tval, output, mask=mask, t_lim=t_lim, second_threshold_mask=second_level, surf=surf, clobber=clobber)

if __name__ == "__main__":
    main()
