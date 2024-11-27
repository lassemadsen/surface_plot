import copy
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import seaborn as sns

import numpy as np
import scipy
from brainspace.mesh.mesh_io import read_surface
from .surface_rendering import render_surface, combine_figures
from .config import get_surface

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarnings 

def plot_pval(pval, output, tval=None, p_threshold=0.01, mask=None, cbar_loc='left', titles=['Positive', 'Negative'], second_threshold_mask=None, expand_edge=True, views='compact', dpi=300, clobber=False):
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
        If None, the p-values are plottet without information about positive or negative t-values
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
    second_threshold_mask : dict or None | None
        If dict: Dictionary with keys "left" and "right", containing data array of cluster mask at 2nd threshold level (e.g. p<0.001)
        Clusters are outlined with a white line on the plot.
    expand_edge : boolean | True
        If True, the white 2nd threshold cluster line is expanded by one vertices for better visuzaliation
    views : str | compact
        Can be either 'standard', 'compact' or 'complete'.
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

    vlim = [0, p_threshold]
    clim_plot = [-(p_threshold / 15), p_threshold] # -(p_threshold/15) to avoid very dark red (jet_r) and get a better looking surface
    cmap = 'turbo_r'

    if tval is None:
        pval_plot = threshold_pmap(pval, p_threshold, tval)

        if second_threshold_mask is not None:
            pval_plot = find_edges(pval_plot, second_threshold_mask, vlim[0]-0.5, -1, expand_edge) # Set edge_val below vmin but above vmin-1 (this will be displayed as white on plot)

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
            tmp_file = f'{tmp_dir}/pval.png'
            render_surface(pval_plot, tmp_file, vlim=vlim, clim=clim_plot, cmap=cmap, dpi=dpi, views=views)

            # Add colorbar
            combine_figures(tmp_file, output, titles=titles, cbArgs=cbar_args, dpi=dpi, clobber=clobber, views=views)

    else:
        posneg_pval = threshold_pmap(pval, p_threshold, tval)
        
        if second_threshold_mask is not None:
            for posneg in ['pos', 'neg']:
                posneg_pval[posneg] = find_edges(posneg_pval[posneg], second_threshold_mask, vlim[0]-0.5, -1, expand_edge) # Set edge_val below vmin but above vmin-1 (this will be displayed as white on plot)
                # OBS: This will plot threshold mask on both pos and neg. Needs to be fixed! 

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
                outfile_posneg = f'{tmp_dir}/{posneg}.png'
                pval_files.append(outfile_posneg)

                render_surface(posneg_pval[posneg], outfile_posneg, mask=mask, vlim=vlim, clim=clim_plot, cmap=cmap, dpi=dpi)
                posneg_figs.append(outfile_posneg)

            # Combine pos and neg figure and add title and combined colorbar
            combine_figures(posneg_figs, output, titles=titles, cbArgs=cbar_args, dpi=dpi, clobber=clobber)
    
    if 'tmp' not in output:
        logger.info(f'{output} saved.')


def plot_tval(tval, output, t_lim=None, t_threshold=2.5, cluster_mask=None, mask=None, p_threshold=None, pval=None, df=None, title=None, cbar_loc='left', second_threshold_mask=None, expand_edge=False, plot_discrete=False, ticks='minmax', views='compact', dpi=300, clobber=False):
    """Plot tval statistics on surface
    Will plot t-values between thresholds
    If p_threshold and df is set, the thresholds are calculated based on the corresponding p-value. 

    Parameters
    ----------
    tval : dict
        Dictionary with keys "left" and "right", containing data array of t-values to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    t_lim : [float, float] | None
        Color lmits of tmap. If None, the min and max values are used (symmetrical around zero)
    t_threshold : float | 2.5
        Treshold of tmap. Values between -threshold;threshold are displayed as white. 
        If 0, entire tmap is plottet
    cluster_mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside cluster mask (indicating surviving clusters)
        Vertices outside cluster_mask will not be marked on the t-value map even if a vertex is above threshold 
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
    title : str | None
        Title of plot
        If None, no title is added.
    cbar_loc : 'left', 'bottom' or None | 'left'
        Location of colorbar
    second_threshold_mask : dict or None | None
        If dict: Dictionary with keys "left" and "right", containing data array of cluster mask at 2nd threshold level (e.g. p<0.001)
        Clusters are outlined with a white line on the plot.
    plot_discrete : boolean | False
        Option to plot surviving clusters in a discrete manner. Only used when second_threshold_mask is set. Plots positve and negative suviving clusters in 4 discrete colors (red: positve, blue: negative). 
    views : str | compact
        Can be either 'standard', 'compact' or 'complete'.
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

    if plot_discrete:
       t_lim = [-5, 5]
       vlim = t_lim
    elif t_lim is None:
        t_min_abs = np.abs(np.min([np.nanmin(tval['left']), np.nanmin(tval['right'])]))
        t_max = np.max([np.nanmax(tval['left']), np.nanmax(tval['right'])])

        t_limit = round(np.max([t_min_abs, t_max]),1)

        if t_limit == 0:
            t_limit = 1

        t_lim = [-t_limit, t_limit]
        vlim = t_lim
    else:
        vlim = t_lim
    
    # Make sure at least one decimal point (for apperance on cbar)
    vlim[0] = vlim[0] + 0.0
    vlim[1] = vlim[1] + 0.0

    cmap = 'RdBu_r'

    if p_threshold is not None:
        if pval is not None:
            tval_thresholded, t_threshold = threshold_tmap(tval, vlim, p_threshold=p_threshold, pval=pval, cluster_mask=cluster_mask)
        elif df is not None:
            tval_thresholded, t_threshold = threshold_tmap(tval, vlim, p_threshold=p_threshold, df=df, cluster_mask=cluster_mask)
        else:
            logger.error('pval or df is not set!')
            raise Exception('pval or df is not set!')
    else:
        tval_thresholded, t_threshold = threshold_tmap(tval, vlim, t_threshold=t_threshold, cluster_mask=cluster_mask)

    if second_threshold_mask is not None:
        if plot_discrete:
            for hemisphere in ['left', 'right']:
                # Assign discrete value to clusters surving fist threshold 
                tval_thresholded[hemisphere][(tval_thresholded[hemisphere] > vlim[0]) & (tval_thresholded[hemisphere] < 0)] = np.linspace(vlim[0],vlim[1],5)[1]
                tval_thresholded[hemisphere][(tval_thresholded[hemisphere] < vlim[1]) & (tval_thresholded[hemisphere] > 0)] = np.linspace(vlim[0],vlim[1],5)[3]

                # Assign discrete value to clusters surving second threshold 
                tval_thresholded[hemisphere][(second_threshold_mask[hemisphere]) & (tval[hemisphere] < 0)] = np.linspace(vlim[0],vlim[1],5)[0] + 0.1 # + 0.1 To aviod background
                tval_thresholded[hemisphere][(second_threshold_mask[hemisphere]) & (tval[hemisphere] > 0)] = np.linspace(vlim[0],vlim[1],5)[4] - 0.1 # - 0.1 To aviod background

            tval_thresholded = find_edges(tval_thresholded, second_threshold_mask, 0, vlim[0]-1, expand_edge=False) # Set edge_val below vmin but above vmin-1 (this will be displayed as white on plot)
        else:
            tval_thresholded = find_edges(tval_thresholded, second_threshold_mask, vlim[0]-0.5, vlim[0]-1, expand_edge) # Set edge_val below vmin but above vmin-1 (this will be displayed as white on plot)

    # Setup colorbar and titles
    if cbar_loc == None:
        cbar_args = None
    elif cbar_loc == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
        cbar_args = {'clim': vlim, 'title': 'T-value', 'fz_title': 24, 'fz_ticks': 24, 'cmap': cmap, 'position': cbar_loc}
    else:
        cbar_args = {'clim': vlim, 'title': 'T-value', 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}

    if plot_discrete:
        cbar_args['n_discrete'] = 4
        cbar_args['title'] = 'Cluster threshold'

    with TemporaryDirectory() as tmp_dir:
        tmp_file = f'{tmp_dir}/tval.png'
        render_surface(tval_thresholded, tmp_file, mask=mask, vlim=vlim, clim=vlim, cmap=cmap, dpi=dpi, views=views)

        if ticks == 'complete':
            ticks=np.array([vlim[0], -t_threshold, t_threshold, vlim[1]])

        # Add colorbar
        combine_figures(tmp_file, output, cbArgs=cbar_args, titles=title, clobber=clobber, ticks=ticks, dpi=dpi, views=views)

    if 'tmp' not in output:
        logger.info(f'{output} saved.')


def threshold_tmap(tval, t_lim, t_threshold=None, p_threshold=None, pval=None, df=None, cluster_mask=None):
    """Threshold t-map
    Differet options (listed in order of priority it mulitple options are given):

    t_threshold: 
        Threshold tval at +/- t_threshold
    p_threshold:
        Threshold tval at +/- p_threshold given the corresponding p values
    pval : array of pval to threshold 
    df : degrees of freedom. Used to caluculate critical t-value 
    cluster_mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside cluster mask (indicating surviving clusters)
        Vertices outside cluster_mask will not be marked on the t-value map even if a vertex is above threshold 
        
    """
    tval = copy.deepcopy(tval) # Copy to avoid overwritting existing tval
    
    if cluster_mask is not None:
        tval = {'left': tval['left']*cluster_mask['left'], 'right': tval['right']*cluster_mask['right']}

    for hemisphere in ['left', 'right']:
        tval[hemisphere] = np.clip(tval[hemisphere], t_lim[0]+1e-3, t_lim[1]-1e-3) # +/- 1e-3 to avoid limits being plottet as "above"/"under" (see render_surface)

        if t_threshold is not None:
            tval[hemisphere][abs(tval[hemisphere]) < t_threshold] = t_lim[0]-1 # Set above t_lim[1] (vmax) to be plottet as white
        elif p_threshold is not None:
            if pval is not None:
                tval[hemisphere][pval[hemisphere] > p_threshold] = t_lim[0]-1 # Set above t_lim[1] (vmax) to be plottet as white
            elif df is not None:
                t_threshold = round(scipy.stats.t.ppf(1-p_threshold, df), 2)
                tval[hemisphere][abs(tval[hemisphere]) < t_threshold] = t_lim[0]-1 # Set above t_lim[1] (vmax) to be plottet as white

    return tval, t_threshold


def threshold_pmap(pval, p_threshold, tval):
    """Threshold p-map

    Parameters
    ----------

    """
    if tval is None:
        pval_return = {'left': copy.deepcopy(pval['left']), 'right': copy.deepcopy(pval['right'])}

        for hemisphere in ['left', 'right']:
            pval_return[hemisphere][pval[hemisphere] > p_threshold] = -1
    else:
        pval_return = {'pos': {'left': copy.deepcopy(pval['left']), 'right': copy.deepcopy(pval['right'])},
                       'neg': {'left': copy.deepcopy(pval['left']), 'right': copy.deepcopy(pval['right'])}} # Copy to aviod overwritting existing pval
        for hemisphere in ['left', 'right']:
            pval_return['pos'][hemisphere][tval[hemisphere] < 0] = -1
            pval_return['pos'][hemisphere][pval_return['pos'][hemisphere] > p_threshold] = -1

            pval_return['neg'][hemisphere][tval[hemisphere] > 0] = -1
            pval_return['neg'][hemisphere][pval_return['neg'][hemisphere] > p_threshold] = -1
    
    return pval_return

def find_edges(data, mask, edge_val, bg_val, expand_edge=True):
    """Find edges of second_threshold mask.

    If edges does not overlap with data (i.e. if edges are in backgound (bg_val)), they are not displayed.
    
    Parameters
    ----------
    
    """
    surface = get_surface(len(data['left']), len(data['right']))

    for hemisphere in ['left', 'right']:

        edge_index = [] 

        # Copy data for current hemisphere and convert to np.ndarray
        tmp_mask = copy.deepcopy(mask[hemisphere]+0).ravel()

        surf = read_surface(surface[hemisphere])

        faces = surf.polys2D 
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

        # --- Expand edge ---
        if expand_edge:
            cluster_indexes = set(vert_idx[tmp_mask == 0])
            expand = set()
            for idx in edge_index:
                neighbours = set(faces[(faces == idx).any(axis=1)].ravel())
                expand.update(neighbours & cluster_indexes)

            edge_index.extend(list(expand))

        # Remove edge_indices if data=bg_val
        edge_index = list(set(np.where((data[hemisphere] != bg_val))[0]) & set(edge_index))

        data[hemisphere][edge_index] = edge_val

    return data