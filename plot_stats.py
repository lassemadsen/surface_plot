import copy
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy

from .surface_rendering import render_surface, combine_figures

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarnings 

def plot_pval(pval, output, tval=None, p_threshold=0.01, mask=None, cbar_loc='left', titles=['Positive', 'Negative'], clobber=False):
    """Plot pval statistics on surface
    If tval is given: it will plot p-values below p_threshold with positive t-values and p-values below p_threshold with negative t-values on seperate plots. 

    Parameters
    ----------
    pval : dict
        Dictionary with keys "left" and "right", containing data array of p-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    tval : None or dict | None
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    p_threshold : float | 0.01
        P value to threshold the statistical map. Default is p<0.01
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
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
    """
    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    vlim = [0, p_threshold]
    clim_plot = [-(p_threshold / 15), p_threshold] # -(p_threshold/15) to avoid very dark red (jet_r) and get a better looking surface
    cmap = 'turbo_r'

    if tval is None:
        # Setup colorbar and titles
        if cbar_loc == None:
            cbar_args = None
        else:
            cbar_args = {'clim': vlim, 'title': 'P-value', 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}


        if titles is not None and len(titles) == 1:
            titles = titles
        else:
            titles = None

        with TemporaryDirectory() as tmp_dir:
            tmp_file = '{}/pval.png'.format(tmp_dir)
            render_surface(pval, tmp_file, vlim=vlim, clim=clim_plot, cmap=cmap)

            # Add colorbar
            combine_figures(tmp_file, output, titles=titles, cbArgs=cbar_args, clobber=clobber)

    else:
        posneg_pval = threshold_pmap(pval, tval)

        pval_files = []

        # Setup colorbar and titles
        if cbar_loc == None:
            cbar_args = None
        elif cbar_loc == 'left':
            cbar_args = {'clim': vlim, 'title': 'P-value', 'fz_title': 11, 'fz_ticks': 11, 'cmap': cmap, 'position': cbar_loc}
        else:
            cbar_args = {'clim': vlim, 'title': 'P-value', 'fz_title': 14, 'fz_ticks': 14, 'cmap': cmap, 'position': cbar_loc}

        with TemporaryDirectory() as tmp_dir:
            posneg_figs = []
            for posneg in ['pos', 'neg']:
                outfile_posneg = '{}/{}.png'.format(tmp_dir, posneg)
                pval_files.append(outfile_posneg)

                render_surface(posneg_pval[posneg], outfile_posneg, mask=mask, vlim=vlim, clim=clim_plot, cmap=cmap)
                posneg_figs.append(outfile_posneg)

            # Combine pos and neg figure and add title and combined colorbar
            combine_figures(posneg_figs, output, titles=titles, cbArgs=cbar_args, clobber=clobber)
    
    if 'tmp' not in output:
        logger.info(f'{output} saved.')


def plot_tval(tval, output, t_lim=None, t_threshold=2.5, mask=None, p_threshold=None, pval=None, df=None, two_tailed=True, title=None, cbar_loc='left', second_threshold_mask=None, surf=None, clobber=False):
    """Plot tval statistics on surface
    Will plot t-values between thresholds
    If p_threshold and df is set, the thresholds are calculated based on the corresponding p-value. 

    Parameters
    ----------
    tval_left : dict
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    t_lim : [float, float] | None
        Color lmits of tmap. If None, the min and max values are used (symmetrical around zero)
    t_threshold : float | 2.5
        Treshold of tmap. Values between -threshold;threshold are displayed as white. 
        If 0, entire tmap is plottet
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
    p_threshold : float | None
        If set, threshold is ignored and the plot is thresholded at the corresponding p-value threshold.
        Note: Either pval or df needs to be set
    pval : dict | None
        Dictionary with keys "left" and "right", containing data array of p-values to plot for left and right hemisphere (without header, i.e. number of vertices)
        Only used if p_threshold is set to threshold corresponding t-values.
        Ignored if p_threshold is None
    df : int | None
        Degrees of freedom.
        Only used if p_threshold is set to calculate conversion between t-values and p-values
        If pval is not None, df is ignored.
        Ignored if p_threshold is None
    two_tailed : boolean | True
        Determines two- or one-tailed t-test. Only used when p_threhold and df is set.
        Ignored if p_threshold or df is None
    title : str | None
        Title of plot
        If None, no title is added.
    cbar_loc : 'left', 'bottom' or None | 'left'
        Location of colorbar
    clobber : Boolean | False
        If true, existing files will be overwritten 

    Notes
    -----
    Colormap could be changed to colormaps found in matplotlib:
    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if t_lim is None:
        t_min_abs = np.abs(np.min([np.nanmin(tval['left']), np.nanmin(tval['right'])]))
        t_max = np.max([np.nanmax(tval['left']), np.nanmax(tval['right'])])

        t_limit = round(np.max([t_min_abs, t_max]),1)

        t_lim = [-t_limit, t_limit]
        vlim = t_lim
    else:
        vlim = t_lim
    
    # Make sure at least one decimal point (for apperance on cbar)
    vlim[0] = vlim[0] + 0.0
    vlim[1] = vlim[1] + 0.0

    cmap = 'RdYlBu_r'

    if p_threshold is not None:
        if pval is not None:
            tval_thresholded = threshold_tmap(tval, vlim, p_threshold=p_threshold, pval=pval)
        elif df is not None:
            tval_thresholded = threshold_tmap(tval, vlim, p_threshold=p_threshold, df=df, two_tailed=two_tailed)
        else:
            logger.error('pval or df is not set!')
            raise Exception('pval or df is not set!')
    else:
        tval_thresholded = threshold_tmap(tval, vlim, t_threshold=t_threshold)

    if second_threshold_mask is not None:
        tval_thresholded = find_edges(tval_thresholded, second_threshold_mask, surf, vlim[0]-1) # Set edge_val above vmax to display as white

    # Setup colorbar and titles
    if cbar_loc == None:
        cbar_args = None
    elif cbar_loc == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
        cbar_args = {'clim': vlim, 'title': 'T-value', 'fz_title': 24, 'fz_ticks': 24, 'cmap': cmap, 'position': cbar_loc}
    else:
        cbar_args = {'clim': vlim, 'title': 'T-value', 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}

    with TemporaryDirectory() as tmp_dir:
        tmp_file = '{}/tval.png'.format(tmp_dir)
        render_surface(tval_thresholded, tmp_file, mask=mask, vlim=vlim, clim=vlim, cmap=cmap)

        # Add colorbar
        combine_figures(tmp_file, output, cbArgs=cbar_args, titles=title, clobber=clobber)

    if 'tmp' not in output:
        logger.info(f'{output} saved.')


def threshold_tmap(tval, t_lim, t_threshold=None, p_threshold=None, pval=None, df=None, two_tailed=None):
    """Threshold t-map
    Differet options (listed in order of priority it mulitple options are given):

    t_threshold: 
        Threshold tval at +/- t_threshold
    p_threshold:
        Threshold tval at +/- p_threshold given the corresponding p values

        pval : array of pval to threshold 
        df : degrees of freedom. Used to caluculate critical t-value 
            two_tailed is used to determine the critical t-value
        
    """
    tval = copy.deepcopy(tval) # Copy to avoid overwritting existing tval

    for hemisphere in ['left', 'right']:
        tval[hemisphere] = np.clip(tval[hemisphere], t_lim[0]+1e-3, t_lim[1]-1e-3) # +/- 1e-3 to avoid limits being plottet as "above"/"under" (see render_surface)

        if t_threshold is not None:
            tval[hemisphere][abs(tval[hemisphere]) < t_threshold] = t_lim[1]+1 # Set above t_lim[1] (vmax) to be plottet as white
        elif p_threshold is not None:
            if pval is not None:
                tval[hemisphere][pval[hemisphere] > p_threshold] = t_lim[1]+1 # Set above t_lim[1] (vmax) to be plottet as white
            elif df is not None:
                if two_tailed:
                    t_critical = scipy.stats.t.ppf(1-p_threshold/2, df)
                else:
                    t_critical = scipy.stats.t.ppf(1-p_threshold, df)
                tval[hemisphere][abs(tval[hemisphere]) < t_critical] = t_lim[1]+1 # Set above t_lim[1] (vmax) to be plottet as white

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

def find_edges(data, mask, surf, edge_val):
    """Find edges
    
    Parameters
    ----------
    
    """
    for hemisphere in ['left', 'right']:

        edge_index = [] 

        # Copy data for current hemisphere and convert to np.ndarray
        tmp_mask = copy.deepcopy(mask[hemisphere]+0).ravel()

        faces = surf[hemisphere].polys2D 
        vert_idx = np.arange(faces.max() + 1)

        # --- Threshold data ---
        not_used_indexes = set(vert_idx[tmp_mask == 1])

        # --- Only use faces containing not_used indexes ---
        faces = faces[np.isin(faces, list(not_used_indexes)).any(axis=1)]

        # --- Find edges ---
        while not_used_indexes:
            current_index = not_used_indexes.pop()
            neighbours = set(faces[(faces == current_index).any(axis=1)].ravel())

            if (tmp_mask[list(neighbours)] == 0).any(axis=0):
                edge_index.append(current_index)

        data[hemisphere][0][edge_index] = edge_val

    return data