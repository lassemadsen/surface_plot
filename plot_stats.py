import copy
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .surface_rendering import render_surface, combine_figures

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarnings 


def plot_stats(pval, tval, output, p_threshold=0.05, plot_tvalue=False, t_lim=None, cbar_loc='left', titles=['Positive', 'Negative'], clobber=False):
    """Plot statistics on surface
    Will plot p-values below p_threshold with positive t-values and p-values below p_threshold with negative t-values.
    If plot_tvalue is set to True, it will produce a single plot with of t-values (with corresponding p-values below p_threshold)

    Parameters
    ----------
    pval : dict
        Dictionary with keys "left" and "right", containing data array of p-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    tval_left : dict
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    p_threshold : float | 0.05
        P value to threshold the statistical map. Default is p<0.05
    plot_tvalue : Boolean | False
        If true, a map of the tvalues will be plottet. 
    t_lim : tuple [tmin, tmax] | None
        Range of tvalues.
        If none, min and max tvalues in the data will be used
    cbar_loc : 'left', 'bottom' or None | 'left'
        Location of colorbar
    titles : tuple | ['Positive', 'Negative']
        Titles of the figures 
        Note: Only used when plotting p-values
    clobber : Boolean | False
        If true, existing files will be overwritten 

    Notes
    -----
    Colormap could be changed to colormaps found in matplotlib:
    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    Functionallity suggestions
    --------------------------
     - Plot only positive or negative t values
    """
    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if plot_tvalue:
        # -- Plot t-values  --
        if t_lim is None:
            t_min_abs = np.abs(np.min([np.min(tval['left']), np.min(tval['right'])]))
            t_max = np.max([np.max(tval['left']), np.max(tval['right'])])

            t_limit = round(np.max([t_min_abs, t_max]),1)

            t_lim = [-t_limit, t_limit]
            clim = t_lim
        else:
            clim = t_lim

        cmap = 'RdYlBu_r'

        tval = threshold_tmap(pval, tval, p_threshold, t_lim)

        # Setup colorbar and titles
        if cbar_loc == None:
            cbar_args = None
        elif cbar_loc == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
            cbar_args = {'clim': clim, 'title': 'T-value', 'fz_title': 24, 'fz_ticks': 24, 'cmap': cmap, 'position': cbar_loc}
        else:
            cbar_args = {'clim': clim, 'title': 'T-value', 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}

        if titles is not None and len(titles) == 1:
            titles = titles
        else:
            titles = None

        with TemporaryDirectory() as tmp_dir:
            tmp_file = '{}/tval.png'.format(tmp_dir)
            render_surface(tval, tmp_file, clim=clim, vlim=clim, cmap=cmap)

            # Add colorbar
            combine_figures(tmp_file, output, titles=titles, cbArgs=cbar_args, clobber=clobber)

    else:
        # -- Plot p-values  --
        clim = [0, p_threshold]
        clim_plot = [-(p_threshold / 15), p_threshold] # -(p_threshold/15) to avoid very dark red (jet_r) and get a better looking surface
        cmap = 'turbo_r'

        posneg_pval = threshold_pmap(pval, tval)

        pval_files = []

        # Setup colorbar and titles
        if cbar_loc == None:
            cbar_args = None
        elif cbar_loc == 'left':
            cbar_args = {'clim': clim, 'title': 'P-value', 'fz_title': 11, 'fz_ticks': 11, 'cmap': cmap, 'position': cbar_loc}
        else:
            cbar_args = {'clim': clim, 'title': 'P-value', 'fz_title': 14, 'fz_ticks': 14, 'cmap': cmap, 'position': cbar_loc}

        with TemporaryDirectory() as tmp_dir:
            posneg_figs = []
            for posneg in ['pos', 'neg']:
                outfile_posneg = '{}/{}.png'.format(tmp_dir, posneg)
                pval_files.append(outfile_posneg)

                render_surface(posneg_pval[posneg], outfile_posneg, clim=clim_plot, vlim=clim, cmap=cmap)
                posneg_figs.append(outfile_posneg)

            # Combine pos and neg figure and add title and combined colorbar
            combine_figures(posneg_figs, output, titles=titles, cbArgs=cbar_args, clobber=clobber)
    
    if 'tmp' not in output:
        logger.info(f'{output} saved.')


def threshold_tmap(pval, tval, threshold, t_lim):
    """Threshold t-map
    """
    tval = copy.deepcopy(tval) # Copy to avoid overwritting existing tval

    for hemisphere in ['left', 'right']:
        tval[hemisphere] = np.clip(tval[hemisphere], t_lim[0], t_lim[1])
        tval[hemisphere][pval[hemisphere] > threshold] = 100 # Set to 100 to avoid over being in range of other t values 
    return tval

def threshold_pmap(pval, tval):
    """Threshold p-map
    """

    posneg = {'pos': {'left': copy.deepcopy(pval['left']), 'right': copy.deepcopy(pval['right'])},
              'neg': {'left': copy.deepcopy(pval['left']), 'right': copy.deepcopy(pval['right'])}} # Copy to aviod overwritting existing pval

    for hemisphere in ['left', 'right']:
            posneg['pos'][hemisphere][tval[hemisphere] < 0] = 1
            posneg['neg'][hemisphere][tval[hemisphere] > 0] = 1
    
    return posneg
