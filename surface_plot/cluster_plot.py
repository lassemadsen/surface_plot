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
sns.set(style='whitegrid')
import statsmodels.api as sm

from .plot_surface import plot_surface

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def boxplot(data1, data2, slm, outdir, g1_name, g2_name, param, paired=False, alpha=0.05, cluster_summary=None, clobber=False):

    outdir = f'{outdir}/cluster_details'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    cluster_mask = {'pos': {'left': [], 'right': []},
                    'neg': {'left': [], 'right': []}}
    cluster_threshold = slm['left'].cluster_threshold # Get primary cluster threshold (used for output naming)

    for posneg in ['pos','neg']:
        if posneg == 'pos':
            posneg_idx = 0
        else:
            posneg_idx = 1

        for hemisphere in ['left', 'right']:
            clusids = list(slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < alpha, 'clusid'])

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
                plt.title(title)
                plt.tight_layout()
                plt.savefig(output)
                plt.close()

                # Assign clusid to mask
                cluster_mask[posneg][hemisphere] = np.where(np.isin(slm[hemisphere].P['clusid'][posneg_idx][0], clusids), slm[hemisphere].P['clusid'][posneg_idx][0], 0)
        
        if any(cluster_mask[posneg]['left']) or any(cluster_mask[posneg]['right']):
            tab10_colors = plt.cm.tab10.colors  # Get the base colors from tab10
            custom_cmap = ListedColormap([tab10_colors[i - 1] for i in clusids])
            matplotlib.colormaps.register(custom_cmap, name=f'custom_cmap_{posneg}')
            cmap = f'custom_cmap_{posneg}'
            
            plot_surface(cluster_mask[posneg], f'{outdir}/{posneg}_cluster_{param.replace(" ", "_")}_{cluster_threshold}.jpg', 
                         clip_data=False, cbar_loc='left', cbar_title='Cluster ID', cmap=cmap, vlim=[1, np.max(clusids)], clobber=clobber)


def correlation_plot(slm, indep_data, indep_name, subjects, outdir, hue=None, alpha=0.05, clobber=False):
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
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if (slm['left'].P is None) or (slm['right'].P is None):
        print('Error: Cluster correction has to be run to identify largest cluster. Run SLM with correction="rft"')
        return 

    cluster_mask = {'pos': {'left': np.zeros_like(indep_data['left'].iloc[:,0]), 'right': np.zeros_like(indep_data['right'].iloc[:,0])},
                    'neg': {'left': np.zeros_like(indep_data['left'].iloc[:,0]), 'right': np.zeros_like(indep_data['right'].iloc[:,0])}}

    for posneg in ['pos','neg']:
        if posneg == 'pos':
            posneg_idx = 0
        else:
            posneg_idx = 1
        for hemisphere in ['left', 'right']:

            cluster_pval = slm[hemisphere].P['clus'][posneg_idx]['P'][0] if not slm[hemisphere].P['clus'][posneg_idx]['P'].empty else 1 # Get pval of largest cluster 
            if cluster_pval > alpha:
                print(f'No {posneg} clusters surviving on {hemisphere} hemisphere.')
                continue

            cluster_threshold = slm[hemisphere].cluster_threshold # Get primary cluster threshold (used for output naming)
            cluster_size = slm[hemisphere].P['clus'][posneg_idx]['nverts'][0] # Get nverts for largest cluster 
            predictor_name = slm[hemisphere].model.matrix.columns[1] # Get predictor name (second column name - first is intercept)
            if len(slm[hemisphere].model.matrix.columns) > 2:
                covars = '+'.join(slm[hemisphere].model.matrix.columns[2:])
                output = f'{outdir}/{posneg}_cluster_{hemisphere}_{indep_name.replace(" ", "_")}_{predictor_name.replace(" ", "_")}+{covars}_{cluster_threshold}.png'
            else:
                output = f'{outdir}/{posneg}_cluster_{hemisphere}_{indep_name.replace(" ", "_")}_{predictor_name.replace(" ", "_")}_{cluster_threshold}.png'

            if not clobber:
                if os.path.isfile(output):
                    logger.info(f'{output} already exists... Skipping')
                    continue

            cluster_mask[posneg][hemisphere] = np.copy(slm[hemisphere].P['clusid'][posneg_idx][0])
            # Ensure only surviving clusters are included
            survived_cluster_idx = list(slm[hemisphere].P['clus'][posneg_idx].loc[slm[hemisphere].P['clus'][posneg_idx].P < 0.05, 'clusid'])
            cluster_mask[posneg][hemisphere][~np.isin(cluster_mask[posneg][hemisphere], survived_cluster_idx)] = 0
                    
            cluster_mean = indep_data[hemisphere][slm[hemisphere].P['clusid'][posneg_idx][0] == 1].mean()

            if hue is None:
                plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name]], axis=1).dropna()
                plot_data.columns = [indep_name, predictor_name]
            else:
                plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name], hue.loc[subjects, hue.columns[0]].reset_index(drop=True)], axis=1).dropna()
                plot_data.columns = [indep_name, predictor_name, hue.columns[0]]

            r2 = _get_r2(plot_data[predictor_name], plot_data[indep_name])
            title = f'{indep_name} - {predictor_name}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'

            if hue is None:
                ax = plt.subplots(figsize=(10, 6))
                ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, ci=None, truncate=False, scatter_kws={'s':200}, line_kws={'linewidth':8})
            else:
                ax = sns.lmplot(x=predictor_name, y=indep_name, hue=hue.columns[0], data=plot_data, ci=None, truncate=False, scatter_kws={'s':200}, fit_reg=False, height=10, aspect=1.4, facet_kws={'legend_out': False})
                ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, scatter=False, ax=ax.axes[0, 0], ci=None, line_kws={'linewidth':8}, color='grey')

            major_formatter = FuncFormatter(__format_values)
            ax.yaxis.set_major_formatter(major_formatter)

            ax.set_title(title)
            plt.tight_layout()

            plt.savefig(output, dpi=300)
            logger.info(f'{output} saved')

        if any(cluster_mask[posneg]['left']) or any(cluster_mask[posneg]['right']):
            if posneg == 'neg':
                cmap = 'Blues'
            else:
                cmap = 'Reds'
            plot_surface(cluster_mask[posneg], f'{outdir}/{posneg}_cluster_{indep_name}_{predictor_name}_{cluster_threshold}.jpg', 
                         clip_data=False, cbar_loc=None, cmap=cmap, vlim=[0.5, 1.5], clobber=clobber)


# --- Helper functions ---
# draw line
# https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
def _newline(p1, p2, color=None, linewidth=1, linestyle='-'):
    ax = plt.gca()
    if color is None:
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='tab:red' if p1[1]-p2[1] > 0 else 'tab:green', marker='o', markersize=14)
    else:
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color=color, marker='o', markersize=14)

    l.set_linewidth(linewidth)
    l.set_linestyle(linestyle)
    ax.add_line(l)
    return l

def __format_values(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '%.2f' % x
    if 0 <= np.abs(x) < 1:
        return val_str.replace('0', '', 1)
    else:
        return val_str

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