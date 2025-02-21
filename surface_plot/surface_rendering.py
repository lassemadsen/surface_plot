from distutils.command.config import config
import gc
import logging
import os
from tempfile import TemporaryDirectory
import copy

import numpy as np
import pandas as pd
from PIL import Image
from visbrain.gui import Figure
from visbrain.objects import BrainObj, SceneObj
import matplotlib.pyplot as plt
from .config import get_surface

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def render_surface(data, outfile, surface=None, mask=None, vlim=None, clim=None, cmap='turbo_r', views='compact', dpi=300, clobber=False):
    """Render surface with given input data

    Parameters
    ----------
    data : dict
        Dictionary with keys "left" and "right", with data array to plot for left and right hemisphere (without header, i.e. number of vertices)
    outfile : string
        Location of output file
    surface : dict | None 
        Dictionary with keys "left" and "right", containing location of left and right surface.
        If None, it will look for a surface with correct number of vertices in surface_plot/surface_data (mni_icbm152_t1_tal_nlin_sym_09c_both_smooth.obj)
    vlim : tuple [vmin, vmax] | None
        The threshold limits.
        Values under vmin is set to darkgrey
        Values over vmax is set to white
        If None, (data.min(), data.max()) will be used
    mask : dict
        Dictionary with keys "left" and "right", containing 1 inside mask and 0 outside mask
        Vertices outside mask will plottet as darkgrey
    clim : tuple [cmin, cmax] | None
        The colorbar limits. If None, (data.min(), data.max()) will be used
        instead.

    cmap : string | 'turbo_r'
        The colormap to use.
        Can be a cmap supported by matplotlib:
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    views : string, 'compact', 'standard' or 'complete' | 'compact'
        Select views to show
        'compact' : left, right, left_inv, right_inv is plottet.
        'standard' : front, back, left, right, left_inv, right_inv is plottet.
        'complete' : front, back, left, right, left_inv, right_inv, top and bottom is plottet.

    clobber : Boolean | False
        If True, existing files will be overwritten
    """

    if clobber == False and os.path.exists(outfile):
        logging.info(f'{outfile} exists... Use clobber=True to overwrite')
        return

    # Handle brainObject 
    b_objects = {'left': [], 'right': [], 'both': []}
    if surface is None: 
        surface = get_surface(len(data['left']), len(data['right']))
        for hemisphere in b_objects:
            b_objects[hemisphere] = BrainObj(surface[hemisphere], hemisphere='both', translucent=False)
    else:
        for hemisphere in ['left', 'right']:
            b_objects[hemisphere] = BrainObj(surface[hemisphere], hemisphere='both', translucent=False)
        # Generate "both" surface
        b_objects['both'] = BrainObj('both', vertices=np.concatenate([b_objects['left'].vertices, b_objects['right'].vertices]), 
                                     faces=np.concatenate([b_objects['left'].faces, b_objects['right'].faces + b_objects['left'].vertices.shape[0]]), hemisphere='both', translucent=False)
    
    if vlim is None:
        vmin = np.round(np.min([np.min(data['left']), np.min(data['right'])]),2)
        vmax = np.round(np.max([np.max(data['left']), np.max(data['right'])]),2)

        vlim = [vmin, vmax]

    for hemisphere in ['left', 'right']:
        np.nan_to_num(data[hemisphere], nan=vlim[0]-1, copy=False)
        if isinstance(data[hemisphere], pd.DataFrame):
            data[hemisphere] = data[hemisphere].values.ravel()
        else:
            data[hemisphere] = data[hemisphere].ravel()
        
        if mask is not None: # Set vertices outside mask less than vmin
            data[hemisphere][~mask[hemisphere]] = vlim[1]+1

    plot_data = {'left': data['left'],
                'right': data['right'],
                'both': np.concatenate((data['left'], data['right']))}

    if views == 'standard':
        views = ['fb', 'lr', 'lr_inv']
    elif views == 'compact':
        views = ['lr', 'lr_inv']
    elif views == 'complete':
        views = ['tb', 'fb', 'lr', 'lr_inv']
    else:
        logger.error(f'Views should be "standard" or "all". Given {views}')
        return

    img_files = []

    with TemporaryDirectory() as tmp_dir:
        for plot_view in views:
            plot_dict, zoom = _get_plot_dict(plot_view)

            sc = SceneObj(bgcolor='white', size=(200, 200))

            for plot in list(plot_dict.keys()):
                # Get hemisphere, view and data to be plotted
                hemisphere = plot.split('_')[0]
                view = plot.split('_')[-1]

                # Create Brain Object
                b_obj = BrainObj(hemisphere, b_objects[hemisphere].vertices, b_objects[hemisphere].faces, hemisphere='both', translucent=False)

                # Add activation to Brain Object
                b_obj.add_activation(data=plot_data[hemisphere], cmap=cmap, clim=clim, vmin=vlim[0],
                                     vmax=vlim[1], under='white', over='darkgrey', hemisphere='both',
                                     smoothing_steps=None, hide_under=vlim[0]-0.5)

                sc.add_to_subplot(b_obj, row=plot_dict[plot][0], col=plot_dict[plot][1], zoom=zoom, rotate=view)
                del b_obj 

            tmp_file = f'{tmp_dir}/{plot_view}.png'
            img_files.append(tmp_file)

            if plot_view == 'fb':
                sc.screenshot(tmp_file, print_size=(10.8,10), dpi=dpi, autocrop=True)
            else:
                sc.screenshot(tmp_file, print_size=(10,10), dpi=dpi, autocrop=True)
            del sc
            gc.collect()

        append_images(img_files, outfile, direction='vertical', scale='width', dpi=dpi)

def _get_plot_dict(plot_view):
    """Return dictionary defining the plot view and zoom

    Parameters
    ---------
    plot_view : str
        String defining the view of the surface object
        Can be: 'tb', 'fb', 'lr' or 'lr_i':
        'Top/Bottom', 'Front/Back', 'Left/Right', 'Left/Right, inverse'

    Notes
    -----
    The dictionary keys define the view of the surface, where
    the string before the underscore tells which surface to plot and
    the string after the underscore tells the view.
    The value tells the [row, col] to plot the surface in the visbrain
    scene object.
    """
    if plot_view == 'tb':  # Top, Bottom
        plot_dict = {'both_top': [0, 0],
                     'both_bottom': [0, 1]}
        zoom = 1.2
    elif plot_view == 'fb':  # Front, Back
        plot_dict = {'both_back': [0, 0],
                     'both_front': [0, 1]}
        zoom = 1.2
    elif plot_view == 'lr':  # Left, Right
        plot_dict = {'left_left': [0, 0],
                     'right_right': [0, 1]}
        zoom = 1.05
    elif plot_view == 'lr_inv':  # Left, Right (inside)
        plot_dict = {'right_left': [0, 0],
                     'left_right': [0, 1]}
        zoom = 1.05
    else:
        logger.error(f'Error plotting {plot_view}')
        return

    return plot_dict, zoom

def append_images(image_files, outfile, direction='horizontal', bg_color=(255, 255, 255), alignment='center', dpi=300, scale=None, clobber=False):
    """Appends images in horizontal/vertical direction.

    Parameters
    ----------
    image_files : list
        List of image files locations
    outfile : str
        Location of the output figure
    direction : str | 'horizontal'
        direction of concatenation, 'horizontal' or 'vertical'
    bg_color : (r,g,b) (int) | (255, 255, 255) : white
        Background color
    alignment: str | 'center'
        alignment mode if images need padding;
        'left', 'right', 'top', 'bottom', or 'center'
    dpi : int | 300
        dpi of the output image
        Note: larger dpi will take longer to produce and
        take up more space
    Scale : 'height', 'width' or None | None
        If 'height', all images will be scaled to the new height
        If 'width', all images will be scaled to the new width
    clobber : Boolean | False
        If True, any existing files will be overwritten
    """
    if not clobber:
        if os.path.isfile(outfile):
            logger.info(f'{outfile} already exists... Skipping')
            return

    images = [Image.open(i) for i in image_files]
    
    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    if scale == 'height':
        # All height are resized to max height (new_height). Width are adjusted acordingly
        width_scaled = [0] * len(images)
        for i in range(len(images)):
            width_scaled[i] = round(widths[i]*(new_height/heights[i]))
            images[i] = images[i].resize((width_scaled[i], new_height), Image.ANTIALIAS)
        new_width = sum(width_scaled)
    elif scale == 'width':
        # All widths are resized to max width (new_width). Heights are adjusted acordingly
        height_scaled = [0] * len(images)
        for i in range(len(images)):
            height_scaled[i] = round(heights[i]*(new_width/widths[i]))
            images[i] = images[i].resize((new_width, height_scaled[i]), Image.ANTIALIAS)
        new_height = sum(height_scaled)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction == 'horizontal':
            y = 0
            if alignment == 'center':
                y = int((new_height - im.size[1]) / 2)
            elif alignment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if alignment == 'center':
                x = int((new_width - im.size[0]) / 2)
            elif alignment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    new_im.save(outfile, dpi=(dpi, dpi))
    plt.close('all')

def combine_figures(files, outfile, direction='horizontal', cbArgs=None, titles=None, ylabels=None,
                    fz_ylabel=14, fz_title=14, discrete=False, ticks='complete', views='standard', dpi=300, clobber=False):
    """Combine figures to one plot with possibility of adding colorbar, labels and titles
    Can also be used to add colorbar to one figure

    Parameters
    ----------
    files : list 
        List of image files to be combined 
    direction: str ('horizontal' or 'vertical')
        direction of which the figures will be combined
    outfile: str
        Location of the output image
    cbArgs : dict | None
        A dictionary to setup the colorbar. Possible keys:
        'title': str - colorbar title
        'cmap': str - colormap
        'clim': [float, float], - colorbar limits
        'fz_ticks': int - fontsize of tick labels on colorbar
        'fz_title': int - fontsize of title on colorbar
        'position': str - position of the colorbar
    titles : str/list/tuple | None
        Specify the title of each figure. If titles is None, no title will be added. 
        If titles is a string, the same title will be applied to all figures. 
        If titles is a list/tuple of strings, the strings inside will be used to set 
        the title of each picture independently (must have the same length as files.)
    ylabels : str/list/tuple | None
        Specify the y-axis label of each figure. If ylabels is None, no label will be added.
        If ylabels is a string, the same label will be applied to all figures. 
        If ylabels is a list/tuple of strings, the strings inside will be used to set
        the label of each picture independently (must have the same length as files.)
    fz_title : int | 14
        Font size of titles (if added)
    fz_ylabel : int | 14
        Font size of ylabels (if added)
    discrete : boolean | False
        Plot discrete
        NOT IMPLEMENTED
    ticks : string/int/float/np.ndarray | 'complete'
        Ticks of the colorbar. This parameter is only active if clim
        is defined. Use 'complete' to see only the minimum,
        maximum, vmin and vmax (if defined). Use 'minmax' to only see
        the maximum and minimum. If ticks is a float, an linear
        interpolation between the maximum and minimum will be used.
        Finally, if ticks is a NumPy array, it will be used as colorbar
        ticks directly.
    views : str | standard
        Can be either 'standard', 'compact' or 'complete'.
        Only important if a colorbar is added to a single plot. 
    dpi : int | 300
        dpi of output image
        Note: larger dpi will take longer to produce and 
        take up more space
    clobber : boolean | False
        If True, any existing files will be overwritten
    """
    if not clobber:
        if os.path.isfile(outfile):
            logger.info(f'{outfile} already exists... Skipping')
            return

    if isinstance(files, str):
        n_fig = 1
    else:
        n_fig = len(files)

    if direction == 'horizontal':
        col = n_fig
        row = 1
    elif direction == 'vertical':
        col = 1
        row = n_fig
    else:
        logger.error(f'Direction parameter is wrong. Should be "vertical" or "horizontal". Got: {direction}')
        return

    # Determine font sizes:
    # The following settings has been optimized to the specific number of images 
    # If another number of images or another number of "views" is plottet, a specific option may need to be added.
    # NB: If only one figure, the fz_title and fz_ylabel is currently 24 no matter the input.
    
    
    if n_fig == 1:
        fz_title = 24
        fz_ylabel = 24

    if cbArgs is not None:
        if cbArgs['position'] == 'left':
            ycb = -25
        else:
            ycb = -10

        # Standard setting. Changed in the follwing if statement if specific option is optimized.
        pltmargin = 0
        height = .5
        width = .007

        if n_fig == 1:
            if views == 'standard': 
                if cbArgs['position'] == 'left':
                    pltmargin = -.07
                    height = .5
                    width = .007
                elif cbArgs['position'] == 'bottom':
                    pltmargin = 0.02
                    height = .4
                    width = .008
                elif cbArgs['position'] == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
                    pltmargin = 0.02
                    height = .4
                    width = .015
                    cbArgs['position'] = 'bottom' # Set position to "correct" position identifiable by f.shared_colorbar 
            if views == 'compact':
                if cbArgs['position'] == 'left':
                    pltmargin = 0.05
                    height = .35
                    width = .009
                elif cbArgs['position'] == 'bottom': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
                    pltmargin = -0.15
                    height = .4
                    width = .008
                elif cbArgs['position'] == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
                    pltmargin = -0.15
                    height = .4
                    width = .015
                    cbArgs['position'] = 'bottom' # Set position to "correct" position identifiable by f.shared_colorbar 
            if views == 'complete':
                if cbArgs['position'] == 'left':
                    pltmargin = -0.30
                    height = .35
                    width = .009
                elif cbArgs['position'] == 'bottom': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
                    pltmargin = 0.02
                    height = .3
                    width = .008
                elif cbArgs['position'] == 'bottom_tval_scaled': # Special scenario were tval is plottet alongside two mean images combined to one (e.g. baseline, followup, tval). Scale cbar accordingly
                    pltmargin = 0.02
                    height = .3
                    width = .008
                    cbArgs['position'] = 'bottom' # Set position to "correct" position identifiable by f.shared_colorbar 
        if n_fig == 2:
            if views == 'standard': 
                if cbArgs['position'] == 'left':
                    pltmargin = .042
                    height = .3
                    width = .007
                elif cbArgs['position'] == 'bottom':
                    pltmargin = -.47
                    height = .4
                    width = .008
            elif views == 'compact':
                if cbArgs['position'] == 'bottom':
                    pltmargin = -.47
                    height = .3
                    width = .007
        if n_fig == 3:
            if cbArgs['position'] == 'bottom':
                pltmargin = -.45
                height = .5
                width = .004

    # Plot figure and shared colorbar (if any)
    f = Figure(files, grid=(row, col), fig_bgcolor='white', titles=titles, figsize=(12, 12),
               ylabels=ylabels, subspace={'hspace': .02, 'wspace': .01, 'top': .85},
               fz_titles=fz_title, fz_ylabels=fz_ylabel, autocrop=True)

    if cbArgs is not None:
        f.shared_colorbar(height=height, width=width, cmap=cbArgs.get('cmap'), fz_title=cbArgs.get('fz_title'), fz_ticks=cbArgs.get('fz_ticks'),
                    pltmargin=pltmargin, ycb=ycb, position=cbArgs.get('position'), title=cbArgs.get('title'), vmin=cbArgs.get('clim')[0],
                    vmax=cbArgs.get('clim')[1], clim=cbArgs.get('clim'), ticks=ticks, n_discrete=cbArgs.get('n_discrete'))

    f.save(outfile, dpi=dpi)
    plt.close('all')
