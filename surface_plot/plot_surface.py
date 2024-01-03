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

def plot_surface(data, output, vlim=None, mask=None, cbar_loc='left', cbar_title='Mean', title=None, cmap='turbo', clobber=False, dpi=300, clip_data=True):
    """Plot data on surface

    Parameters
    ----------
    data : dict
        Dictionary with keys "left" and "right", containing data array of data to plot for left and right hemisphere (without header, i.e. number of vertices)
    output : str
        Location to save output
    vlim : [min, max] | None
        Value limits on the plot.
        If None, the min and max values will be used
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
    cbar_loc : 'left', 'bottom' or None | 'left'
        Location of colorbar
    cbar_title : str | 'Mean'
        Title on colorbar
    title : str | None
        Title of the figure
    cmap: str | 'RdYlBu_r'
        Colormap used for plotting
        Recommendations:
        'RdBu_r' is good for t values
        'turbo' otherwise
    clobber : Boolean | False
        If true, existing files will be overwritten
    clip_data : Boolean | True
        If true, data is clipped to vlim, else values outside vlim will be plottet as white (under) or gray (over)

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

    plot_data = copy.deepcopy(data)

    if vlim is None:
        vmin = np.round(np.min([np.min(data['left']), np.min(data['right'])]),1)
        vmax = np.round(np.max([np.max(data['left']), np.max(data['right'])]),1)

        vlim = [vmin, vmax]

    # Make sure at least one decimal point (for apperance on cbar)
    vlim[0] = vlim[0] + 0.0
    vlim[1] = vlim[1] + 0.0

    # Clip data to min and max values
    if clip_data:
        plot_data['left'] = np.clip(plot_data['left'], vlim[0]+1e-3, vlim[1]-1e-3) # 1e-6 to avoid rounding error when plotting (cliped vertices might be seen as out of range)
        plot_data['right'] = np.clip(plot_data['right'], vlim[0]+1e-3, vlim[1]-1e-3)

    # Setup colorbar and titles
    if cbar_loc == None:
        cbar_args = None
    else:
        cbar_args = {'clim': vlim, 'title': cbar_title, 'fz_title': 16, 'fz_ticks': 16, 'cmap': cmap, 'position': cbar_loc}

    with TemporaryDirectory() as tmp_dir:
        tmp_file = f'{tmp_dir}/data.png'
        render_surface(plot_data, tmp_file, mask=mask, vlim=vlim, clim=vlim, cmap=cmap, dpi=dpi)

        # Add colorbar
        combine_figures(tmp_file, output, titles=title, cbArgs=cbar_args, clobber=clobber, dpi=dpi)

    if 'tmp' not in output:
        logger.info(f'{output} saved.')

