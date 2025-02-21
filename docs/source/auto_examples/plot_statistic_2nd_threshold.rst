
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_statistic_2nd_threshold.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_statistic_2nd_threshold.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_statistic_2nd_threshold.py:


Tutorial 04: Plot of statistical maps with two thresholds
=========================================================
Example script to plot statistical maps with two different statistical thresholds.
The first threshold is plottet as normal, the second threshold is outlined by a white line.

The second_level data is defined by a mask containing e.g. surviving clusters (1's)

Note: the cortical surface template is needed to segment the clusters. 

.. GENERATED FROM PYTHON SOURCE LINES 11-38







.. code-block:: Python

    from surface_plot import plot_stats
    import numpy as np

    outdir = 'data/second_level/test_figures'
    clobber = True
    mask = {'left': [], 'right': []}

    # Load data
    tval_left = np.loadtxt('https://www.dropbox.com/s/5ek63zf5l2iwd8f/tval_left.csv?dl=1')
    tval_right = np.loadtxt('https://www.dropbox.com/s/def320uevw5ivsb/tval_right.csv?dl=1')
    second_level_left = np.loadtxt('https://www.dropbox.com/s/y93z65g2jx0zs2x/second_level_left.csv?dl=1')
    second_level_right = np.loadtxt('https://www.dropbox.com/s/wgp0j0gp35oe41g/second_level_right.csv?dl=1')

    tval = {'left': tval_left,
            'right': tval_right}

    second_level = {'left': second_level_left,
                    'right': second_level_right}

    t_lim = [-5, 5]

    mask['left'] = ~np.isnan(tval['left'])
    mask['right'] = ~np.isnan(tval['right'])

    output = f'{outdir}/second_level.pdf'

    plot_stats.plot_tval(tval, output, mask=mask, t_lim=t_lim, second_threshold_mask=second_level, expand_edge=True, clobber=clobber)


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 25.120 seconds)


.. _sphx_glr_download_auto_examples_plot_statistic_2nd_threshold.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_statistic_2nd_threshold.ipynb <plot_statistic_2nd_threshold.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_statistic_2nd_threshold.py <plot_statistic_2nd_threshold.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_statistic_2nd_threshold.zip <plot_statistic_2nd_threshold.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
