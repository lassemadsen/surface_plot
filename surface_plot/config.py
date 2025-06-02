import pathlib
import sys
# Define surface to use

SURFACE_MNI = {'left': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii',
               'right': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii',
               'both': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_both_smooth.gii'}

N_VERTEX_MNI = {'left': 81349,
                'right': 81233}

def get_surface(nv_left, nv_right):
    """Return the surface template, that fits with the number of vertices in the left and right hemisphere

    Parameters
    ----------
    nv_left : int
        Number of vertices in left hemishpere
    nv_right : int
        Number of vertices in right hemishpere
    """

    if nv_left == N_VERTEX_MNI['left'] and nv_right == N_VERTEX_MNI['right']:
        surface = SURFACE_MNI
    else:
        sys.exit('Error: Number of vertices do not match any surface...')

    return surface