import pathlib
# Define surface to use
SURFACE = {'left': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii',
           'right': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii',
           'both': f'{pathlib.Path(__file__).parent.resolve()}/surface_data/mni_icbm152_t1_tal_nlin_sym_09c_both_smooth.gii',}