import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.2)
sns.set(style='white')
import statsmodels.api as sm

from .plot_surface import plot_surface

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def boxplot(data1, data2, slm, outdir, g1_name, g2_name, param, paired=False, alpha=0.05, cluster_summary=None, clobber=False):
    """
    Generates and saves boxplots for significant clusters in brain surface data.

    Parameters
    ----------
    data1 : dict
        Dictionary containing group 1 data with keys ('left', 'right').
    data2 : dict
        Dictionary containing group 2 data with keys ('left', 'right').
    slm : dict
        Statistical results from brainstat containing cluster information.
    outdir : str
        Output directory path for saving plots.
    g1_name : str
        Name of the first group.
    g2_name : str
        Name of the second group.
    param : str
        Name of the parameter being analyzed.
    paired : bool (optional) | False
        Whether the data is paired.
    alpha : float | 0.05
        Corrected p-value threshold on cluster-level (family wise error rate)
    cluster_summary : DataFrame (optional) | None
        DataFrame with cluster area and anatomical location details.
    clobber : bool (optional) | False
        If False, skips generating plots if they already exist.

    Returns:
    - Saves boxplot figures for each significant cluster.
    """
    outdir = f'{outdir}/cluster_details'
    
    cluster_mask = {'pos': {'left': [], 'right': []},
                    'neg': {'left': [], 'right': []}}
    cluster_threshold = slm['left'].cluster_threshold # Get primary cluster threshold (used for output naming)

    for posneg in ['pos','neg']:
        if posneg == 'pos':
            posneg_idx = 0
        else:
            posneg_idx = 1
        
        clusids_list = []

        for hemisphere in ['left', 'right']:
            clusids = list(slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < alpha, 'clusid'])

            if clusids:
                Path(outdir).mkdir(parents=True, exist_ok=True) # Only create folder if there are surviving clusters
                clusids_list.extend(clusids)

            for clusid in clusids:
                cluster_pval = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'P'].values[0]
                
                output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{param.replace(" ", "_")}_{cluster_threshold}.pdf'

                # Set title
                if cluster_summary is None:
                    cluster_size = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'nverts'].values[0] # Get number of vertices in cluster
                    title = f'{param}, {g1_name} - {g2_name}. Cluster {clusid}.\n{hemisphere} hemisphere, N vertices={cluster_size:.0f}, FWE p-value={cluster_pval:.1e}'
                else:
                    cluster_size = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid), 'Cluster area (mm2)'].values[0]
                    cluster_location = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid), 'Anatomical location (peak)'].values[0]

                    title = f'{param}, {g1_name} - {g2_name}. Cluster {clusid}.\n{cluster_location} - {hemisphere},' + rf' Size: {cluster_size} mm$^2$, FWE p-value={cluster_pval:.1e}'

                if not clobber:
                    if os.path.isfile(output):
                        logger.info(f'{output} already exists... Skipping')
                        continue
            
                # Get mean of cluster in each group
                cluster_mean_g1 = data1[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == clusid].mean().to_frame(name=param)
                cluster_mean_g1['group'] = g1_name
                cluster_mean_g2 = data2[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == clusid].mean().to_frame(name=param)
                cluster_mean_g2['group'] = g2_name

                # Plot boxplot
                plot_data = pd.concat([cluster_mean_g1, cluster_mean_g2])

                plt.figure()
                sns.boxplot(x='group', y=param, data=plot_data, hue='group')
                if paired:
                    for i in range(len(cluster_mean_g1)):
                        plt.plot([0, 1], [cluster_mean_g1.iloc[i, 0], cluster_mean_g2.iloc[i, 0]], color='gray', linestyle='--', linewidth=1)
                    sns.stripplot(x='group', y=param, data=plot_data, jitter=False, color='black', size=5, dodge=True)
                else:
                    sns.swarmplot(x='group', y=param, data=plot_data, hue='group', edgecolor='black', linewidth=1, size=5, legend=False)
                plt.title(title, size=10)
                plt.tight_layout()
                plt.savefig(output)
                plt.close()

                # Assign clusid to mask
                cluster_mask[posneg][hemisphere] = np.where(np.isin(slm[hemisphere].P['clusid'][posneg_idx][0], clusids), slm[hemisphere].P['clusid'][posneg_idx][0], 0)
        
        clusids_list = list(set(clusids_list)) # Make sure colormap is based on both hemispheres.
        if clusids_list:
            tab10_colors = plt.cm.tab10.colors  # Get the base colors from tab10
            custom_cmap = ListedColormap([tab10_colors[i - 1] for i in clusids_list])
            matplotlib.colormaps.register(custom_cmap, name='custom_cmap')

            if np.max(clusids_list) == 1:
                vlim = [0.9, 1.1]
            else:
                vlim = [1, np.max(clusids_list)]
            
            plot_surface(cluster_mask[posneg], f'{outdir}/{posneg}_cluster_{param.replace(" ", "_")}_{cluster_threshold}.jpg', 
                         clip_data=False, cbar_loc='left', cbar_title='Cluster ID', cmap='custom_cmap', vlim=vlim, clobber=clobber)
            
            matplotlib.colormaps.unregister('custom_cmap')


def correlation_plot(slm, indep_data, indep_name, subjects, outdir, hue=None, alpha=0.05, cluster_summary=None, clobber=False):
    """
    
    Parameters
    ----------
    slm : dict['left', 'right']
        Dictionary with keys "left" and "right", containing results from brainstat SLM
    indep_data : pd.DataFrame - dict{'left', 'right'}
        Dataframe of surface data
        Dictionary with keys "left" and "right", containing surface data of the independent variable for left and right hemisphere
    indep_name : str
        Name of independent surface data
    subjects : list
        List of subjects (same as indices in indep_data)
    outdir : str
        Location of ouputs
    hue : dataframe
        Dataframe with subject ids as index and group as column
    alpha : float | 0.05
        Corrected p-value threshold on cluster-level (family wise error rate)
    clobber : Boolean | False
        If true, existing files will be overwritten
    """
    outdir = f'{outdir}/cluster_details'

    cluster_mask = {'pos': {'left': [], 'right': []},
                    'neg': {'left': [], 'right': []}}
    cluster_threshold = slm['left'].cluster_threshold # Get primary cluster threshold (used for output naming)
    predictor_name = slm['left'].model.matrix.columns[1] # Get predictor name (second column name - first is intercept)

    for posneg in ['pos','neg']:
        if posneg == 'pos':
            posneg_idx = 0
        else:
            posneg_idx = 1

        clusids_list = []

        for hemisphere in ['left', 'right']:
            clusids = list(slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < alpha, 'clusid'])

            if clusids:
                Path(outdir).mkdir(parents=True, exist_ok=True) # Only create folder if there are surviving clusters
                clusids_list.extend(clusids)

            for clusid in clusids:
                cluster_pval = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'P'].values[0]

                if len(slm[hemisphere].model.matrix.columns) > 2:
                    covars = '+'.join(slm[hemisphere].model.matrix.columns[2:])
                    output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{indep_name.replace(" ", "_")}_{predictor_name.replace(" ", "_")}+{covars}_{cluster_threshold}.pdf'
                else:
                    output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{indep_name.replace(" ", "_")}_{predictor_name.replace(" ", "_")}_{cluster_threshold}.pdf'

                if not clobber:
                    if os.path.isfile(output):
                        logger.info(f'{output} already exists... Skipping')
                        continue
                    
                cluster_mean = indep_data[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == 1].mean()

                if hue is None:
                    plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name]], axis=1).dropna()
                    plot_data.columns = [indep_name, predictor_name]
                else:
                    plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name], hue.loc[subjects, hue.columns[0]].reset_index(drop=True)], axis=1).dropna()
                    plot_data.columns = [indep_name, predictor_name, hue.columns[0]]

                r2 = _get_r2(plot_data[predictor_name], plot_data[indep_name])
                # Set title
                if cluster_summary is None:
                    cluster_size = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'nverts'].values[0] # Get number of vertices in cluster
                    title = f'{indep_name} - {predictor_name}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'
                else:
                    cluster_size = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid), 'Cluster area (mm2)'].values[0]
                    cluster_location = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid), 'Anatomical location (peak)'].values[0]
                    title = f'{indep_name} - {predictor_name}. Cluster {clusid}.\n{cluster_location} - {hemisphere},' + rf' Size: {cluster_size} mm$^2$, FWE p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'

                if hue is None:
                    ax = plt.subplots(figsize=(10, 6))
                    ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, ci=None, truncate=False, scatter_kws={'s':100}, line_kws={'linewidth':5, 'alpha': 0.8})
                else:
                    ax = sns.lmplot(x=predictor_name, y=indep_name, hue=hue.columns[0], data=plot_data, ci=None, truncate=False, scatter_kws={'s':100}, fit_reg=False, height=10, aspect=1.4, facet_kws={'legend_out': False})
                    ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, scatter=False, ax=ax.axes[0, 0], ci=None, line_kws={'linewidth':5, 'alpha':0.8}, color='grey')

                ax.set_title(title, size=10)
                plt.tight_layout()
                plt.savefig(output)
                plt.close()
                
                # Assign clusid to mask
                cluster_mask[posneg][hemisphere] = np.where(np.isin(slm[hemisphere].P['clusid'][posneg_idx][0], clusids), slm[hemisphere].P['clusid'][posneg_idx][0], 0)

        clusids_list = list(set(clusids_list)) # Make sure colormap is based on both hemispheres.

        if clusids_list:
            tab10_colors = plt.cm.tab10.colors  # Get the base colors from tab10
            custom_cmap = ListedColormap([tab10_colors[i - 1] for i in clusids_list])
            matplotlib.colormaps.register(custom_cmap, name=f'custom_cmap_{posneg}', force=True)
            cmap = f'custom_cmap_{posneg}'

            if np.max(clusids_list) == 1:
                vlim = [0.9, 1.1]
            else:
                vlim = [1, np.max(clusids_list)]
            
            plot_surface(cluster_mask[posneg], f'{outdir}/{posneg}_cluster_{indep_name.replace(" ", "_")}_{predictor_name.replace(" ", "_")}_{cluster_threshold}.jpg', 
                         clip_data=False, cbar_loc='left', cbar_title='Cluster ID', cmap=cmap, vlim=vlim, clobber=clobber)


def _get_r2(x, y):
    """Calculate r squared

    Return
    ------
    r2 : R squared value
    """
    X = sm.add_constant(x)
    result = sm.OLS(y, X).fit()
    r2 = result.rsquared

    return r2