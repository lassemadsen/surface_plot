from functools import cmp_to_key
import logging
import os
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from .plot_stats import plot_pval, plot_tval
from .surface_rendering import render_surface, combine_figures, append_images

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_mean_stats(mean_group1, mean_group2, tval, output, plot_tvalue=False, pval=None, t_threshold=2.5, df=None, p_threshold=0.01, mask=None, vlim_mean=None, mean_titles=None, stats_titles=None, cb_mean_title='Mean', t_lim=None, second_threshold_mask=None, expand_edge=True, dpi=300, clobber=False):
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
    tval_left : dict
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    plot_tvalue : Boolean | False
        If true, a map of the tvalues will be plottet. 
    pval : dict | None
        Dictionary with keys "left" and "right", containing data array of p-values to plot for left and right hemisphere (without header, i.e. number of vertices)
        Only used if p_threshold is set to threshold corresponding t-values.
        Ignored if p_threshold is None
    t_threshold : float | 2.5
        Treshold of tmap. Values between -threshold;threshold are displayed as white. 
        If 0, entire tmap is plottet
    df : int | None
        Degrees of freedom.
        Only used if p_threshold is set to calculate conversion between t-values and p-values
        If pval is not None, df is ignored.
        Ignored if plot_tvalue is true
    p_threshold : float | 0.01
        If set, threshold is ignored and the plot is thresholded at the corresponding p-value threshold.
        Note: Either pval or df needs to be set
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
    vlim_mean : tuple [min, max] | [0,1]
        Limits of the mean plots 
    mean_titles : list | None
        Titles of the two mean plots : ['title1', 'title2']. If None, no title is added
    stats_titles : list | None
        Titles of the stats plot. E.g. ['Positive', 'Negative'] or ['Change']
    cb_mean_title : str | 'Mean'
        Title on the colorbar for the mean plots
    t_lim : tuple [tmin, tmax] | None
        Range of tvalues.
        If none, min and max tvalues in the data will be used
    second_threshold_mask : dict or None | None
        If dict: Dictionary with keys "left" and "right", containing data array of cluster mask at 2nd threshold level (e.g. p<0.001)
        Clusters are outlined with a white line on the plot.
    expand_edge : boolean | True
        If True, the white 2nd threshold cluster line is expanded by one vertices for better visuzaliation
    clobber : Boolean | False
        If true, existing files will be overwritten 

    Notes
    -----
    Colormap could be changed to colormaps found in matplotlib:
    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    """
    if not clobber:
        if os.path.isfile(output):
            logger.info(f'{output} already exists... Skipping')
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if isinstance(mean_titles, list) & isinstance(stats_titles, list) or mean_titles == stats_titles == None:
        pass
    else: 
        logger.warning('Titles not given for both mean and stats. Plot might look weird..')
    
    if vlim_mean is None:
        mean_min = round(min(np.nanpercentile(mean_group1['left'], 0.5), np.nanpercentile(mean_group1['right'], 0.5), np.nanpercentile(mean_group2['left'], 0.5), np.nanpercentile(mean_group2['right'], 0.5)),2)
        mean_max = round(max(np.nanpercentile(mean_group1['left'], 99.5), np.nanpercentile(mean_group1['right'], 99.5), np.nanpercentile(mean_group2['left'], 99.5), np.nanpercentile(mean_group2['right'], 99.5)),2)
        vlim_mean = [mean_min, mean_max]

    for hemisphere in ['left', 'right']:
        mean_group1[hemisphere] = np.clip(mean_group1[hemisphere], vlim_mean[0]+1e-3, vlim_mean[1]-1e-3)
        mean_group2[hemisphere] = np.clip(mean_group2[hemisphere], vlim_mean[0]+1e-3, vlim_mean[1]-1e-3)

    with TemporaryDirectory() as tmp_dir:
        cmap = 'turbo'
        # Plot mean group1
        tmp_mean1 = f'{tmp_dir}/mean1.png'
        render_surface(mean_group1, tmp_mean1, vlim=vlim_mean, clim=vlim_mean, mask=mask, cmap=cmap, dpi=dpi)

        # Plot mean group2
        tmp_mean2 = f'{tmp_dir}/mean2.png'
        render_surface(mean_group2, tmp_mean2, vlim=vlim_mean, clim=vlim_mean, mask=mask, cmap=cmap, dpi=dpi)

        # Combine means with shared colorbar - Setup colorbar
        cbar_args = {'clim': vlim_mean, 'title': cb_mean_title, 'fz_title': 14, 'fz_ticks': 14, 'cmap': cmap, 'position': 'bottom'}

        tmp_mean = f'{tmp_dir}/mean.png'
        combine_figures([tmp_mean1, tmp_mean2], tmp_mean, cbArgs=cbar_args, titles=mean_titles, dpi=dpi)

        # Plot stats
        tmp_stats = f'{tmp_dir}/stats.png'

        # Setup colorbar
        if plot_tvalue:
            cbar_loc = 'bottom_tval_scaled' # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
            plot_tval(tval, tmp_stats, t_lim=t_lim, t_threshold=t_threshold, mask=mask, pval=pval, p_threshold=p_threshold, df=df, title=stats_titles, cbar_loc=cbar_loc, second_threshold_mask=second_threshold_mask, expand_edge=expand_edge, dpi=dpi, clobber=clobber)
        else:
            if pval is None:
                logger.error('Pval needs to be set. Otherwise set plot_tvalue=True')
                return
            cbar_loc = 'bottom'
            plot_pval(pval, tmp_stats, tval=tval, p_threshold=p_threshold, mask=mask, cbar_loc=cbar_loc, titles=mean_titles, second_threshold_mask=second_threshold_mask, expand_edge=expand_edge, dpi=dpi, clobber=clobber)

        # Combine to one plot 
        append_images([tmp_mean, tmp_stats], output, direction='horizontal', scale='height', dpi=dpi, clobber=True)
    
    logger.info(f'{output} saved')

