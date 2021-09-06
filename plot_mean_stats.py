import copy
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import surface_rendering as sr
import plot_stats as ps

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger('plot_mean_stats')

def plot_mean_stats(mean_group1, mean_group2, pval, tval, output, p_threshold=0.05, vlim_mean=[0,1], mean_titles=None, stats_titles=None, cb_mean_title='Mean', plot_tvalue=False, t_lim=None, clobber=False):
    """Plot mean and statistics on surface
    Will plot mean of group 1 and mean of group 2 along with p-values or t-values.
    Will plot p-values below p_threshold with positive t-values and p-values below p_threshold with negative t-values.
    If plot_tvalue is set to True, it will produce a single plot with of t-values (with corresponding p-values below p_threshold)
    Colorbar location will be at the bottom

    Parameters
    ----------
    mean_group1 : dict
        Dictionary with keys "left" and "right", containing data array of the mean data for the first group to plot for left and right hemisphere (without header, i.e. number of vertices)
    mean_group2 : dict
        Dictionary with keys "left" and "right", containing data array of the mean data for the second group to plot for left and right hemisphere (without header, i.e. number of vertices)
    pval : dict
        Dictionary with keys "left" and "right", containing data array of p-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    tval_left : dict
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    p_threshold : float | 0.05
        P value to threshold the statistical map. Default is p<0.05
    vlim_mean : tuple [min, max] | [0,1]
        Limits of the mean plots 
    mean_titles : list | None
        Titles of the two mean plots : ['title1', 'title2']. If None, no title is added
    stats_titles : list | None
        Titles of the stats plot. E.g. ['Positive', 'Negative'] or ['Change']
    cb_mean_title : str | 'Mean'
        Title on the colorbar for the mean plots
    plot_tvalue : Boolean | False
        If true, a map of the tvalues will be plottet. 
    t_lim : tuple [tmin, tmax] | None
        Range of tvalues.
        If none, min and max tvalues in the data will be used
    clobber : Boolean | False
        If true, existing files will be overwritten 

    Notes
    -----
    Colormap could be changed to colormaps found in matplotlib:
    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    Functionallity suggestions
    --------------------------
    """
    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if isinstance(mean_titles, list) & isinstance(stats_titles, list) or mean_titles == stats_titles == None:
        pass
    else: 
        logger.warning('Titles not given for both mean and stats. Plot might look weird..')

    with TemporaryDirectory() as tmp_dir:
        # Plot mean group1
        tmp_mean1 = '{}/mean1.png'.format(tmp_dir)
        sr.render_surface(mean_group1, tmp_mean1, vlim=vlim_mean, clim=vlim_mean)

        # Plot mean group2
        tmp_mean2 = '{}/mean2.png'.format(tmp_dir)
        sr.render_surface(mean_group2, tmp_mean2, vlim=vlim_mean, clim=vlim_mean)

        # Combine means with shared colorbar - Setup colorbar
        cbar_args = {'clim': vlim_mean, 'title': cb_mean_title, 'fz_title': 14, 'fz_ticks': 14, 'cmap': 'turbo', 'position': 'bottom'}

        tmp_mean = '{}/mean.png'.format(tmp_dir)
        sr.combine_figures([tmp_mean1, tmp_mean2], tmp_mean, cbArgs=cbar_args, titles=mean_titles)

        # Plot stats
        tmp_stats = '{}/stats.png'.format(tmp_dir)

        # Setup colorbar
        if plot_tvalue:
            cbar_loc = 'bottom_tval_scaled' # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
        else:
            cbar_loc = 'bottom'

        ps.plot_stats(pval, tval, tmp_stats, p_threshold=p_threshold, plot_tvalue=plot_tvalue, t_lim=t_lim, cbar_loc=cbar_loc, titles=stats_titles)

        # Combine to one plot 
        sr.append_images([tmp_mean, tmp_stats], output, direction='horizontal', scale='height', clobber=True)
    
    logger.info(f'{output} saved')

