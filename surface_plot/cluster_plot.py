import logging
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.2)
sns.set(style='whitegrid')
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def boxplot(data1, data2, slm, outdir, g1_name, g2_name, param, alpha=0.05, clobber=False):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    for hemisphere in ['left', 'right']:
        for posneg in [[0, 'pos'], [1, 'neg']]:
            cluster_pval = slm[hemisphere].P['clus'][posneg[0]]['P'][0] if not slm[hemisphere].P['clus'][posneg[0]]['P'].empty else 1 # Get pval of largest cluster 
            if cluster_pval > alpha:
                continue

            cluster_threshold = slm[hemisphere].cluster_threshold # Get primary cluster threshold (used for output naming)
            cluster_size = slm[hemisphere].P['clus'][posneg[0]]['nverts'][0] # Get nverts for largest cluster 
            title = f'{param}, {g1_name} - {g2_name}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}'
            output = f'{outdir}/{posneg[1]}_cluster_{hemisphere}_{param}_{cluster_threshold}.pdf'

            if not clobber:
                if os.path.isfile(output):
                    logger.info(f'{output} already exists... Skipping')
                    continue
            
            output_cluster_mask = f'{output.split(".pdf")[0]}_clusterMask.csv'
            np.savetxt(output_cluster_mask, slm[hemisphere].P['clusid'][posneg[0]][0])

            cluster_mean_1 = data1[hemisphere][slm[hemisphere].P['clusid'][posneg[0]][0] == 1].mean().to_frame(name=param)
            cluster_mean_1['group'] = g1_name
            cluster_mean_2 = data2[hemisphere][slm[hemisphere].P['clusid'][posneg[0]][0] == 1].mean().to_frame(name=param)
            cluster_mean_2['group'] = g2_name

            plot_data = pd.concat([cluster_mean_1, cluster_mean_2])

            plt.figure()
            sns.boxplot(y=param, x='group', data=plot_data)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output)
            plt.clf()


def slope_plot(slm, data1, data2, categories, param_name, outdir, title=None, clobber=False, alpha=0.05, extra_lines=None, print_id=False):
    """
    
    Parameters
    ----------
    data1 : pd.DataFrame
        Dataframe of first data
    data2 : pd.DataFrame
        Dataframe of second data
    param_name : str
        Name of parameter - used for file naming 
    categories : list of str
        Names of [data1, data2] in correct order
        I.e ['baseline', 'followup'] if paired t-test or ['pib_pos', 'control'] if independent t-test
    """

    Path(outdir).mkdir(parents=True, exist_ok=True)

    if (slm['left'].P is None) or (slm['right'].P is None):
        print('Error: Cluster correction has to be run to identify largest cluster. Run SLM with correction="rft"')
        return 

    for hemisphere in ['left', 'right']:
        for posneg in [[0, 'pos'], [1, 'neg']]:

            cluster_pval = slm[hemisphere].P['clus'][posneg[0]]['P'][0] if not slm[hemisphere].P['clus'][posneg[0]]['P'].empty else 1 # Get pval of largest cluster 
            if cluster_pval > alpha:
                print(f'No {posneg[1]} clusters surviving on {hemisphere} hemisphere.')
                continue

            cluster_threshold = slm[hemisphere].cluster_threshold # Get primary cluster threshold (used for output naming)
            cluster_size = slm[hemisphere].P['clus'][posneg[0]]['nverts'][0] # Get nverts for largest cluster 
            output = f'{outdir}/{param_name}_{posneg[1]}_cluster_{hemisphere}_{categories[0]}_{categories[1]}_{cluster_threshold}.png'

            if not clobber:
                if os.path.isfile(output):
                    logger.info(f'{output} already exists... Skipping')
                    continue

            output_cluster_mask = f'{output.split(".png")[0]}_clusterMask.csv'
            np.savetxt(output_cluster_mask, slm[hemisphere].P['clusid'][posneg[0]][0])
                    
            cluster_mean1 = data1[hemisphere][slm[hemisphere].P['clusid'][posneg[0]][0] == 1].mean()
            cluster_mean2 = data2[hemisphere][slm[hemisphere].P['clusid'][posneg[0]][0] == 1].mean()

            title = f'{param_name}: {categories[0]} - {categories[1]}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}'

            _, ax = plt.subplots(1,1,figsize=(14,14), dpi=80)

            # Vertical Lines
            ymin = min(min(cluster_mean1.values), min(cluster_mean2.values))  
            ymax = max(max(cluster_mean1.values), max(cluster_mean2.values))
            yrange = ymax-ymin
            ymin = ymin - 0.1*yrange
            ymax = ymax + 0.1*yrange

            ax.vlines(x=1, ymin=ymin, ymax=ymax, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
            ax.vlines(x=2, ymin=ymin, ymax=ymax, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

            # Points
            ax.scatter(y=cluster_mean1.values, x=np.repeat(1, cluster_mean1.shape[0]), s=10, color='black', alpha=0.7)
            ax.scatter(y=cluster_mean2.values, x=np.repeat(2, cluster_mean2.shape[0]), s=10, color='black', alpha=0.7)

            for p1, p2, c in zip(cluster_mean1.values, cluster_mean2.values, data1[hemisphere].columns):
                _newline([1,p1], [2,p2])
                if print_id:
                    ax.text(1-0.05, p1, c, horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
                # ax.text(3+0.05, p2, c + ', ' + str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size':14})

            if extra_lines is not None:
                colors = ['tab:blue', 'deepskyblue', 'steelblue']
                color_idx = 0
                for key in extra_lines:
                    _newline([1,extra_lines[key][0]], [2,extra_lines[key][1]], linewidth=3, linestyle='-', color=colors[color_idx])
                    color_idx = (color_idx + 1) % len(colors) # Cycle through colors
                    ax.text(0.98, extra_lines[key][0], key, horizontalalignment='right', verticalalignment='center', fontdict={'size':14})

            # Decoration
            ax.set_title(title, fontdict={'size':22})
            ax.set(xlim=(.9,2.1), ylim=(ymin,ymax))
            ax.set_ylabel('Mean', fontsize=18)
            ax.set_xticks([1,2])
            ax.set_xticklabels([categories[0], categories[1]], fontdict={'size':20})
            plt.yticks(fontsize=18)

            # Lighten borders
            plt.gca().spines["top"].set_alpha(.0)
            plt.gca().spines["bottom"].set_alpha(.0)
            plt.gca().spines["right"].set_alpha(.0)
            plt.gca().spines["left"].set_alpha(.0)
            plt.savefig(output, dpi=300)
            plt.close()
            logger.info(f'{output} saved')

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

    for hemisphere in ['left', 'right']:
        for posneg in [[0, 'pos'], [1, 'neg']]:

            cluster_pval = slm[hemisphere].P['clus'][posneg[0]]['P'][0] if not slm[hemisphere].P['clus'][posneg[0]]['P'].empty else 1 # Get pval of largest cluster 
            if cluster_pval > alpha:
                print(f'No {posneg[1]} clusters surviving on {hemisphere} hemisphere.')
                continue

            cluster_threshold = slm[hemisphere].cluster_threshold # Get primary cluster threshold (used for output naming)
            cluster_size = slm[hemisphere].P['clus'][posneg[0]]['nverts'][0] # Get nverts for largest cluster 
            predictor_name = slm[hemisphere].model.matrix.columns[1] # Get predictor name (second column name - first is intercept)
            if len(slm[hemisphere].model.matrix.columns) > 2:
                covars = '+'.join(slm[hemisphere].model.matrix.columns[2:])
                output = f'{outdir}/{posneg[1]}_cluster_{hemisphere}_{indep_name}_{predictor_name}+{covars}_{cluster_threshold}.png'
            else:
                output = f'{outdir}/{posneg[1]}_cluster_{hemisphere}_{indep_name}_{predictor_name}_{cluster_threshold}.png'

            if not clobber:
                if os.path.isfile(output):
                    logger.info(f'{output} already exists... Skipping')
                    continue

            output_cluster_mask = f'{output.split(".pdf")[0]}_clusterMask.csv'
            np.savetxt(output_cluster_mask, slm[hemisphere].P['clusid'][posneg[0]][0])
                    
            cluster_mean = indep_data[hemisphere][slm[hemisphere].P['clusid'][posneg[0]][0] == 1].mean()

            if hue is None:
                plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name]], axis=1).dropna()
                plot_data.columns = [indep_name, predictor_name]
            else:
                plot_data = pd.concat([cluster_mean[subjects].reset_index(drop=True), slm[hemisphere].model.matrix[predictor_name], hue.loc[subjects, hue.columns[0]].reset_index(drop=True)], axis=1).dropna()
                plot_data.columns = [indep_name, predictor_name, hue.columns[0]]

            r2 = _get_r2(plot_data[predictor_name], plot_data[indep_name])
            title = f'{indep_name} - {predictor_name}, {hemisphere} hemisphere\nN vertices={cluster_size:.0f}, corrected cluster p-value={cluster_pval:.1e}, $R^2$: {r2:.2f}'

            if hue is None:
                ax = plt.subplots(figsize=(10, 8))
                ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, ci=None, truncate=False, scatter_kws={'s':90}, line_kws={'linewidth':5})
            else:
                ax = sns.lmplot(x=predictor_name, y=indep_name, hue=hue.columns[0], data=plot_data, ci=None, truncate=False, scatter_kws={'s':140}, fit_reg=False, height=10, aspect=1.2, facet_kws={'legend_out': False})
                ax = sns.regplot(x=predictor_name, y=indep_name, data=plot_data, scatter=False, ax=ax.axes[0, 0], ci=None, line_kws={'linewidth':5}, color='grey')

            major_formatter = FuncFormatter(__format_values)
            ax.yaxis.set_major_formatter(major_formatter)

            # ax.grid(b=True, which='major', color='w', linewidth=1.0)
            # ax.grid(b=True, which='minor', color='w', linewidth=0.5)
            ax.set_title(title)
            plt.tight_layout()

            plt.savefig(output, dpi=300)
            logger.info(f'{output} saved')

# --- Helper functions ---
# draw line
# https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
def _newline(p1, p2, color=None, linewidth=1, linestyle='-'):
    ax = plt.gca()
    if color is None:
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='tab:red' if p1[1]-p2[1] > 0 else 'tab:green', marker='o', markersize=6)
    else:
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color=color, marker='o', markersize=6)

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