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
            clusids = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < alpha, 'clusid'].astype(int).tolist()

            if clusids:
                Path(outdir).mkdir(parents=True, exist_ok=True) # Only create folder if there are surviving clusters
                clusids_list.extend(clusids)
            else:
                cluster_mask[posneg][hemisphere] = np.zeros_like(slm[hemisphere].t[0]) # Initiate cluster mask in case of surviving clusters on opposite hemisphere 

            for clusid in clusids:
                cluster_pval = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'P'].values[0]
                
                output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{param.replace(" ", "_")}_p{cluster_threshold}.pdf'
                output_csv = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{param.replace(" ", "_")}_p{cluster_threshold}.csv'

                # Set title
                if cluster_summary is None:
                    cluster_size = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'nverts'].values[0] # Get number of vertices in cluster
                    title = f'{param}, {g1_name} - {g2_name}. Cluster {clusid}.\n{hemisphere} hemisphere, N vertices={cluster_size:.0f}, FWE p-value={cluster_pval:.1e}'
                else:
                    cluster_size = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid) & (cluster_summary.sign_t == posneg), 'Cluster area (mm2)'].values[0]
                    cluster_location = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid) & (cluster_summary.sign_t == posneg), 'Anatomical location (peak)'].values[0]

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
                plot_data.to_csv(output_csv)

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
            num_clusters = len(clusids_list)
            if num_clusters > 10:
                custom_colors = [plt.cm.viridis(i / num_clusters) for i in range(num_clusters)]
                custom_cmap = ListedColormap(custom_colors)
            else:
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


def correlation_plot(slm, dep_data, indep_data, dep_name, indep_name, outdir, quadratic=False, 
                     hue=None, alpha=0.05, cluster_summary=None, clobber=False):
    """
    
    Parameters
    ----------
    slm : dict['left', 'right']
        Dictionary with keys "left" and "right", containing results from brainstat SLM
    dep_data : pd.DataFrame or dict{'left', 'right'}
        If the independent variable is a "regular" variable (e.g. age, MMSE etc.) dep_data should be a 1D pd.Dataframe with dep_name as column name and ids as indices.
        If the independent variable is another surface, it should be similar to indep_data, i.e.:
        Dictionary with keys "left" and "right", containing surface data of the dependent variable for left and right hemisphere
    indep_data : dict{'left': pd.DataFrame, 'right': pd.DataFrame}
        Dataframe of surface data
        Dictionary with keys "left" and "right", containing surface data of the independent variable for left and right hemisphere
    indep_name : str
        Name of independent surface data
    outdir : str
        Location of ouputs
    hue : dataframe | None
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

    # Order of regplot
    if quadratic:
        order = 2
    else:
        order = 1

    for posneg in ['pos','neg']:
        if posneg == 'pos':
            posneg_idx = 0
        else:
            posneg_idx = 1

        clusids_list = []

        for hemisphere in ['left', 'right']:
            clusids = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < alpha, 'clusid'].astype(int).tolist()

            if clusids:
                Path(outdir).mkdir(parents=True, exist_ok=True) # Only create folder if there are surviving clusters
                clusids_list.extend(clusids)
            else:
                cluster_mask[posneg][hemisphere] = np.zeros_like(slm[hemisphere].t[0]) # Initiate cluster mask in case of surviving clusters on opposite hemisphere 

            for clusid in clusids:
                cluster_pval = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'P'].values[0]

                if len(slm[hemisphere].model.matrix.columns) > 2:
                    covars = '+'.join(slm[hemisphere].model.matrix.columns[2:])
                    output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{dep_name.replace(" ", "_")}_{indep_name.replace(" ", "_")}+{covars}_p{cluster_threshold}.pdf'
                    output_csv = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{dep_name.replace(" ", "_")}_{indep_name.replace(" ", "_")}+{covars}_p{cluster_threshold}.csv'
                else:
                    output = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{dep_name.replace(" ", "_")}_{indep_name.replace(" ", "_")}_p{cluster_threshold}.pdf'
                    output_csv = f'{outdir}/{posneg}_cluster{clusid}_{hemisphere}_{dep_name.replace(" ", "_")}_{indep_name.replace(" ", "_")}_p{cluster_threshold}.csv'

                if not clobber:
                    if os.path.isfile(output):
                        logger.info(f'{output} already exists... Skipping')
                        continue
                
                if isinstance(indep_data, dict): # If correlation was performed with another surface
                    cluster_mean_indep_data = indep_data[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == clusid].mean().to_frame(name=indep_name)
                elif isinstance(indep_data, pd.Series) or isinstance(indep_data, pd.DataFrame):
                    cluster_mean_indep_data = indep_data
                else:
                    print('Error. Dep_data should be dict{"left", "right"} or pd.DataFrame')
                    return

                cluster_mean_dep_data = dep_data[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == clusid].mean().to_frame(name=dep_name)
                
                if hue is None:
                    plot_data = pd.concat([cluster_mean_dep_data, cluster_mean_indep_data], axis=1).dropna()
                else:
                    plot_data = pd.concat([cluster_mean_dep_data, cluster_mean_indep_data, hue], axis=1).dropna()

                plot_data.to_csv(output_csv)

                r2 = _get_r2(plot_data[dep_name], plot_data[indep_name], quadratic=quadratic)
                # Set title
                if cluster_summary is None:
                    cluster_size = slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].clusid == clusid, 'nverts'].values[0] # Get number of vertices in cluster
                    title = f'{dep_name} - {indep_name}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'
                else:
                    cluster_size = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid) & (cluster_summary.sign_t == posneg), 'Cluster area (mm2)'].values[0]
                    cluster_location = cluster_summary.loc[(cluster_summary.Hemisphere == hemisphere) & (cluster_summary.clusid == clusid) & (cluster_summary.sign_t == posneg), 'Anatomical location (peak)'].values[0]
                    title = f'{dep_name} - {indep_name}. Cluster {clusid}.\n{cluster_location} - {hemisphere},' + rf' Size: {cluster_size} mm$^2$, FWE p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'

                if hue is None:
                    ax = plt.subplots(figsize=(10, 6))
                    ax = sns.regplot(x=indep_name, y=dep_name, data=plot_data, ci=None, truncate=False, scatter_kws={'s':100}, line_kws={'linewidth':5, 'alpha': 0.8}, order=order)
                else:
                    ax = sns.lmplot(x=indep_name, y=dep_name, hue=hue.columns[0], data=plot_data, ci=None, truncate=False, scatter_kws={'s':100}, fit_reg=False, height=10, aspect=1.4, facet_kws={'legend_out': False})
                    ax = sns.regplot(x=indep_name, y=dep_name, data=plot_data, scatter=False, ax=ax.axes[0, 0], ci=None, line_kws={'linewidth':5, 'alpha':0.8}, color='grey', order=order)

                ax.set_title(title, size=10)
                plt.tight_layout()
                plt.savefig(output)
                plt.close()
                
                # Assign clusid to mask
                cluster_mask[posneg][hemisphere] = np.where(np.isin(slm[hemisphere].P['clusid'][posneg_idx][0], clusids), slm[hemisphere].P['clusid'][posneg_idx][0], 0)

        clusids_list = list(set(clusids_list)) # Make sure colormap is based on both hemispheres.

        if clusids_list:
            num_clusters = len(clusids_list)
            if num_clusters > 10:
                custom_colors = [plt.cm.viridis(i / num_clusters) for i in range(num_clusters)]
                custom_cmap = ListedColormap(custom_colors)
            else:
                tab10_colors = plt.cm.tab10.colors  # Get the base colors from tab10
                custom_cmap = ListedColormap([tab10_colors[i - 1] for i in clusids_list])

            matplotlib.colormaps.register(custom_cmap, name='custom_cmap')

            if np.max(clusids_list) == 1:
                vlim = [0.9, 1.1]
            else:
                vlim = [1, np.max(clusids_list)]
            
            plot_surface(cluster_mask[posneg], f'{outdir}/{posneg}_cluster_{dep_name.replace(" ", "_")}_{indep_name.replace(" ", "_")}_{cluster_threshold}.jpg', 
                         clip_data=False, cbar_loc='left', cbar_title='Cluster ID', cmap='custom_cmap', vlim=vlim, clobber=clobber)
            matplotlib.colormaps.unregister('custom_cmap')


def _get_r2(x, y, quadratic=False):
    """
    Calculate R squared.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Predictor variable.
    y : array-like, shape (n_samples,)
        Response variable.
    quadratic : bool, default=False
        If True, include x^2 term in the model.

    Returns
    -------
    r2 : float
        R squared value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if quadratic:
        x_quad = x ** 2
        X = np.column_stack((x, x_quad))
    else:
        X = x

    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    return result.rsquared