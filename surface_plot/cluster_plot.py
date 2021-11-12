import logging
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def slope_plot(data1, data2, cluster_mask, categories, output, title=None, clobber=False, extra_lines=None):
    """
    
    Parameters
    ----------
    data1 : pd.DataFrame
        Dataframe of first data
    data2 : pd.DataFrame
        Dataframe of second data
    cluster_mask : array
        Array defining the cluster to be plottet (mean value within cluster is plottet)
    categories : list of str
        Names of [data1, data2] in correct order
        I.e ['baseline', 'followup'] if paired t-test or ['pib_pos', 'control'] if independent t-test
    """
    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if isinstance(cluster_mask, np.ndarray):
        mask = pd.DataFrame(cluster_mask.T)
    else:
        print('Cluster mask should be np.array')

    mean1 = data1[mask[0]==1].mean()
    mean2 = data2[mask[0]==1].mean()

    title = f'{title}\nSize: {sum(mask[0] == 1)}'

    data1 = pd.DataFrame(data={'mean': mean1})
    data2 = pd.DataFrame(data={'mean': mean2})

    _, ax = plt.subplots(1,1,figsize=(14,14), dpi=80)

    # Vertical Lines
    ymin = min(min(mean1.values), min(mean2.values))  
    ymax = max(max(mean1.values), max(mean2.values))
    yrange = ymax-ymin
    ymin = ymin - 0.1*yrange
    ymax = ymax + 0.1*yrange

    ax.vlines(x=1, ymin=ymin, ymax=ymax, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=2, ymin=ymin, ymax=ymax, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    # Points
    ax.scatter(y=mean1.values, x=np.repeat(1, mean1.shape[0]), s=10, color='black', alpha=0.7)
    ax.scatter(y=mean2.values, x=np.repeat(2, mean2.shape[0]), s=10, color='black', alpha=0.7)

    for p1, p2, c in zip(mean1.values, mean2.values, data1.index):
        _newline([1,p1], [2,p2])
        # ax.text(1-0.05, p1, c + ', ' + str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
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
    logger.info('{} saved'.format(output))

def correlation_plot(data1, predictor, cluster_mask, categories, output, title=None, clobber=False, plot_simple=True):
    """
    
    Parameters
    ----------
    data1 : pd.DataFrame
        Dataframe of surface data
    predictor : pd.DataFrame
        data of predictor 
    cluster_mask : array
        Array defining the cluster to be plottet (mean value within cluster is plottet)
    categories : list of str
        Names of [data1, data2] in correct order
        I.e ['PiB suvr', 'MMSE']
    """

    if not clobber:
        if os.path.isfile(output):
            logger.info('{} already exists... Skipping'.format(output))
            return

    outdir = '/'.join(output.split('/')[:-1])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if isinstance(cluster_mask, np.ndarray):
        mask = pd.DataFrame(cluster_mask.T)
    else:
        print('Cluster mask should be np.array')

    mean1 = data1[mask[0]==1].mean()

    title = f'{title}\nSize: {sum(mask[0] == 1)}'

    plot_data = pd.concat([mean1, predictor], axis=1).dropna()
    plot_data.columns = categories

    if plot_simple:
        sns.set(font_scale=1.5)
        plt.subplots(figsize=(10, 8))
        ax = sns.regplot(x=categories[0], y=categories[1], data=plot_data, ci=None)
    else:
        sns.set(font_scale=1.6)
        plt.subplots(figsize=(10, 8))
        r2 = _get_r2(plot_data[categories[0]], plot_data[categories[1]])
        # label = 'Cluster size: {:.2f} $mm^2$, $p_{{mean}}$: {:.4f}, $R^2$: {:.2f}'.format(cluster_size, mean_pval, r2)
        label = f'$R^2$: {r2:.2}'
        ax = sns.regplot(x=categories[0], y=categories[1], data=plot_data, ci=None, label=label, truncate=False)
        ax.legend(loc="best")

    major_formatter = FuncFormatter(__format_values)
    ax.yaxis.set_major_formatter(major_formatter)

    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax.set_title(title)
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.tight_layout()

    plt.savefig(output, dpi=300)
    logger.info('{} saved'.format(output))


        # major_formatter = FuncFormatter(my_formatter)
        # ax.yaxis.set_major_formatter(major_formatter)

        # ax.grid(b=True, which='major', color='w', linewidth=1.0)
        # ax.grid(b=True, which='minor', color='w', linewidth=0.5)
        # ax.set_title(title)
        # plt.ylim(ylim)
        # plt.xlim(xlim)
        # plt.tight_layout()

        # plt.savefig(outfile, dpi=300)
        # logger.info('{} saved'.format(outfile))


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