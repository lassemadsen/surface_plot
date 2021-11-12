import logging 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import os
from pathlib import Path

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
