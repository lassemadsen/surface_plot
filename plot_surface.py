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

def plot_surface(data, output, mask=None, min_val=None, max_val=None, cbar_loc='left', title=None, cmap='RdYlBu_r', clip_min=True, clip_max=True, clobber=False):
    """Plot data on surface

    Parameters
    ----------
    data : dict
        Dictionary with keys "left" and "right", containing data array of data to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
    min_val = float | None
        Min value to plot. The 'clip_min' option can be used to control values below min_val
        If None, min value of data is used
    max_val = float | None
        Max value to plot. The 'clip_max' option can be used to control values above max_val
        If None, max value of data is used
    cbar_loc : 'left', 'bottom' or None | 'left'
        Location of colorbar
    titles : str | None
        Title of the figure
    cmap: str | 'RdYlBu_r'
        Colormap used for plotting
        Recommendations:
        'RdYlBu_r' is good for t values
        'tubo' otherwise
    clip_min : bool | True
        If True values below min_val are set to min_val (i.e. displayed as same color on plot)
        If False, values below min_val are displayed as white (background). Useful when plotting statistics.
        Only used when min_val is set (not None)
    clip_max : bool | True
        If True values above max_val are set to max_val (i.e. displayed as same color on plot)
        If False, values above max_val are displayed as white (background). Useful when plotting statistics.
        Only used when max_val is set (not None)

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

    plot_data = copy.deepcopy(data) # Copy to avoid changing original data if clipping is performed.
    
    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if max_val is None:
        max_val = round(np.max([np.nanmax(plot_data['left']), np.nanmax(plot_data['right'])]),1)
    else:
        if clip_max:
            plot_data['left'] = np.clip(plot_data['left'], None, np.nextafter(max_val,0))
            plot_data['right'] = np.clip(plot_data['right'], None, np.nextafter(max_val,0))

    if min_val is None:
        min_val = round(np.min([np.nanmin(plot_data['left']), np.nanmin(plot_data['right'])]),1)
    else:
        if clip_min:
            plot_data['left'] = np.clip(plot_data['left'], np.nextafter(min_val,9), None)
            plot_data['right'] = np.clip(plot_data['right'], np.nextafter(min_val,9), None)
        else:
            max_data = np.max((np.nanmax(plot_data['left']), np.nanmax(plot_data['right'])))
            plot_data['left'][plot_data['left'] < min_val] = max_data + 1 # Set above max to be display as white on plot (dependent on render_surface)
            plot_data['right'][plot_data['right'] < min_val] = max_data + 1 # Set above max to be display as white on plot (dependent on render_surface)


    lim = [min_val, max_val]

    # Setup colorbar and titles
    if cbar_loc == None:
        cbar_args = None
    elif cbar_loc == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
        cbar_args = {'clim': lim, 'title': 'T-value', 'fz_title': 24, 'fz_ticks': 24, 'cmap': cmap, 'position': cbar_loc}
    else:
        cbar_args = {'clim': lim, 'title': 'T-value', 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}

    with TemporaryDirectory() as tmp_dir:
        tmp_file = '{}/tval.png'.format(tmp_dir)
        render_surface(plot_data, tmp_file, mask=mask, vlim=lim, clim=lim, cmap=cmap)

        # Add colorbar
        combine_figures(tmp_file, output, titles=title, cbArgs=cbar_args, clobber=clobber)

    if 'tmp' not in output:
        logger.info(f'{output} saved.')

