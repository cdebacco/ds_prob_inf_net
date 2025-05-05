
"""
Functions for plotting results.
**Copyright**: Caterina De Bacco, 2025
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tools as tl
import scipy.stats as st

from probinet.visualization.plot import extract_bridge_properties

sns.set_style('white')

# plt.style.available
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-v0_8-notebook')
plt.rcParams['font.family'] = 'PT Sans'
# plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 12
edge_color = "darkgrey"

default_colors = {'green_forest':'#1E995B','green_aqua':'#66d8ba',
				  'green_sb':'#99E3CF','green_sb_dark':'#709B8F',
				  'blue':'#697cd4','blue_royal':'#2658C6','blue_aqua':'#44BFF9',
				  'blue_sb':'#6C8DC5','blue_sb_dark':'#465B89','blue_sky':'#598AEA',
				  'blue_columbia':'#B0C5EE',
				  'purple':'#aa65b2','purple_sb':'#C598CA','purple_sb_dark':'#9E58A7',
				  'red_adobe':'#A42416','red_salmon':'#F09473','red_salmon_dark':'#EA8079',
				  'yellow_sand':'#EABB5C','black':'black',
				  'light_grey':'#EEEEEE','dark_grey':'#22312b'}


dpi = 100

DEFAULT_COLORS = {0: '#C998D7', 1: '#85C98D', 2: '#F8C38A', 3: '#846C5B', 4:'#95CBC7', 5:'#F5D058',
            6:'#E74E4E', 7:'#CCA5A5', 8:'#6359BE', 9:'#E868B5'}
DEFAULT_EDGE_COLOR = 'darkgrey'
DEFAULT_NODE_COLOR = 'darkorchid'
# Utils to visualize the data

default_colormap = plt.cm.tab20
default_colors = {i: default_colormap(i) for i in range(20)}
default_colors2 = {i+20: plt.cm.tab20c(i) for i in range(20)}
default_colors.update(default_colors2)

default_colors_dict = {'green_forest':'#1E995B','green_aqua':'#66d8ba',
                  'green_sb':'#99E3CF','green_sb_dark':'#709B8F',
                  'blue':'#697cd4','blue_royal':'#2658C6','blue_aqua':'#44BFF9',
                  'blue_sb':'#6C8DC5','blue_sb_dark':'#465B89','blue_sky':'#598AEA',
                  'blue_columbia':'#B0C5EE',
                  'purple':'#aa65b2','purple_sb':'#C598CA','purple_sb_dark':'#9E58A7',
                  'red_adobe':'#A42416','red_salmon':'#F09473','red_salmon_dark':'#EA8079',
                  'yellow_sand':'#EABB5C','black':'black',
                  'light_grey':'#EEEEEE','dark_grey':'#22312b',
                  'red':'#B93B28','yellow_light':'#F3BA46','grey_dark':'#413036',
                    'light_grey':'#F3F3F3','white':'#FFFFFF','blue':'#73B3E6','yellow_desert':'#F3BA46',
                    'blue_dark':'#04143A'
                    }

BLACK = "#282724"

def plot_matrix(
        M_input: np.array,
        cmap='PuBuGn',
        title='',
        labels=None,
        outfile: str = None,
        node_order=None,
        fs: int = 8,
        figsize=(4, 4),
        colorbar=True,
        ax=None,
        vmin: float = None,
        vmax: float = None
        ):

    if M_input.ndim == 2:
        M = np.zeros((1,M_input.shape[0],M_input.shape[1]))
        M[0,:,:] = M_input.copy()
    else:
        M = M_input

    L = M.shape[0]
    K = np.unique(M[0]).shape[0]

    if M[0].dtype == 'int':
        cmap = plt.get_cmap(cmap, K)
    else:
        cmap = 'Blues'

    if vmin is None:
        vmin = np.unique(M).min()
    if vmax is None:
        vmax = np.unique(M).max()

    # if labels is None:
    #     labels = np.arange(M.shape[1])
    cmap = plt.get_cmap(cmap, K)

    for l in range(L):
        N = M[0].shape[0]
        if ax is None:
            print('Plotting in a new figure')
            fig, ax = plt.subplots(figsize=figsize)

        if node_order is not None:
            if M[0].dtype == 'int':
                ax.imshow(M[l][node_order][:, node_order], cmap=cmap, vmin=0 - 0.5, vmax=K - 0.5)
            else:
                ax.imshow(M[l][node_order][:, node_order], cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            ax.imshow(M[l], cmap=cmap,vmin=vmin,vmax=vmax)

        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_title(title, fontsize=fs)
        if labels is not None:
            ax.set_xticks(np.arange(len(labels)), fontsize=fs)
            ax.set_yticks(np.arange(len(labels)), fontsize=fs)
            ax.set_xticklabels(labels, fontsize=fs)
            ax.set_yticklabels(labels, fontsize=fs),
        else:
            plt.xticks([])
            plt.yticks([])

        if M[0].dtype == 'int':
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break

            # Colorbar
            if colorbar:
                divider = make_axes_locatable(ax)
                colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
                cb = plt.colorbar(PCM, ax=ax, cax=colorbar_axes, ticks=np.arange(0, K + 1))
                cb.ax.tick_params(labelsize=fs, length=0)

        if outfile is not None:
            plt.savefig(outfile, dpi=300, format='png', pad_inches=0.1, bbox_inches='tight')
            print(f"Matrix saved in {outfile}")
    return plt

def plot_network(
    data,
    u: np.ndarray,
    nodeLabel2Id: dict = None,
    position: dict = None,
    outdir: str = "../figures/",
    colors: dict = None,
    filename: str = None,
    plot_soft: bool = True,
    node_size: list = None,
    ms: int = 10,
    lecture_id: int = None,
    ax: plt.Axes = None,
    title: str = None,
    plot_labels: bool = True,
    threshold: float = 0.1,
    node_labels: dict = None,
    figsize: tuple = (16,10)
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    if nodeLabel2Id is None:
        nodeLabel2Id = {k: i for i, k in enumerate(data.nodes)}

    if position is None:
        position = tl.get_custom_node_positions(data.graph_list[0])

    if colors is None:
        colors = default_colors

    if node_size is None:
        node_size = [np.log(data.graph_list[0].degree[i]) * ms + 20 for i in data.nodes]

    if node_labels is None:
        node_labels = {}

        for n, d in list(data.graph_list[0].degree()):
            if d > 4: node_labels[n] = n
            if np.count_nonzero(u[nodeLabel2Id[n]]) > 1:
                node_labels[n] = n

    if plot_labels == True:
        nx.draw_networkx_labels(data.graph_list[0], position, font_size=8, alpha=0.8, labels=node_labels, ax=ax)
    nx.draw_networkx_edges(data.graph_list[0], pos=position, width=0.1, ax=ax)

    if title is not None:
        ax.set_title(title)
    plt.axis('off')

    if plot_soft == True:
        # ax = plt.gca()
        for j, n in enumerate(data.graph_list[0].nodes()):
            wedge_sizes, wedge_colors = extract_bridge_properties(j, colors, u, threshold=threshold)
            if len(wedge_sizes) > 0:
                _ = ax.pie(
                    wedge_sizes,
                    center=position[n],
                    colors=wedge_colors,
                    radius=(node_size[j]) * 0.001
                )
                ax.axis("equal")
    else:
        nx.draw_networkx_nodes(data.graph_list[0], position, node_size=node_size,
                               node_color=default_colors['blue'], edgecolors=default_colors['dark_grey'], ax=ax)

    plt.tight_layout()

    filename = tl.get_filename(filename, lecture_id=lecture_id)

    tl.savefig(plt, outfile=filename, outdir=outdir)


def plot_L(values, indices=None, k_i=5, figsize=(7, 7), int_ticks=False, ylab='Log-likelihood', xlab='Iterations'):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab+' values')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()

def plot_net_over(G, pos, u, plt, cm, wedgeprops=None,radius = 0.001, node_size: list = None):
    if wedgeprops is None:
        wedgeprops = {'edgecolor': 'lightgrey'}
    
    if node_size is None:
        node_size = [10 for i in G.nodes()]

    ax = plt.gca()
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color='lightgrey', alpha=0.5)
    for i,n in enumerate(list(G.nodes())):
        
        wedge_sizes, wedge_colors = tl.extract_overlapping_membership(i, cm, u, threshold=0.1)
        if len(wedge_sizes) > 0:
            wedge_sizes /= wedge_sizes.sum()
        if wedge_sizes.sum() > 1:
            # print(f"before: sum = {wedge_sizes.sum()}\n{wedge_sizes}")
            wedge_sizes = np.array([np.round(w,3) for w in wedge_sizes])
            wedge_sizes[0] -= wedge_sizes.sum() -1
            # print(f"after: sum = {wedge_sizes.sum()}\n{wedge_sizes}")
        assert wedge_sizes.sum() <= 1, f"sum = {wedge_sizes.sum()}\n{wedge_sizes}"
        if len(wedge_sizes) > 0:
            # wedge_sizes /= wedge_sizes.sum()
            # assert wedge_sizes.sum() <= 1, f"sum = {wedge_sizes.sum()}\n{wedge_sizes}"
            try:
                _ = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors, radius=(min(100, node_size[i])) * radius,
                                 wedgeprops=wedgeprops, normalize=False)
            except: 1
    ax.axis("equal")


def plot_net_hard(G, pos, node_size=None, plt = None, cm=None, u = None, ms=5, ax = None,
                  plot_labels: bool=False, node_labels: list=None,plot_arrows: bool = False,
                  edge_color: str = 'lightgrey',node_color: str = default_colors_dict['blue_sb_dark']):

    if cm is None:
        cm = plt.cm.tab20
    if ax is None:
        assert plt is not None, f"plt is None!"
        ax = plt.gca()

    if u is None:
        u = np.zeros(G.number_of_nodes()).astype(int)

    if node_size is None:
        node_size = [1 for n in G.nodes()]

    if node_color is None:
        node_color = [cm(u[i]) for i, n in enumerate(G.nodes())]

    if node_labels is not None:
        plot_labels = True

    nx.draw_networkx_edges(G, pos, arrows=plot_arrows, edge_color=edge_color, alpha=0.5,
                           arrowstyle='-|>',connectionstyle="arc3,rad=0.5")
    nx.draw_networkx_nodes(G, pos, node_size=[ms * ns for ns in node_size],
                           node_color=node_color)

    if plot_labels == True:
        if node_labels is None:
            node_labels = G.nodes()
        nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8, labels=node_labels, ax=ax)


    ax.axis("equal")
    ax.axis("off")


def assign_edge_color_from_score(e: tuple, ranks: np.ndarray):
    if ranks[e[0]] > ranks[e[1]]:
        return default_colors_dict['blue_sky']
    if ranks[e[0]] < ranks[e[1]]:
        return default_colors_dict['red_salmon']
    else:
        return default_colors_dict['dark_grey']

def plot_score_network(A: np.ndarray, ranks: np.ndarray,nodeId2Label: dict=None,
                x_jit: float = 0.01,ms: int = 200,seed: int = 10,
                node_size=None, cm=None, u=None, ax=None,
                plot_labels: bool = False, node_labels: list = None, plot_arrows: bool = True,
                 node_color: str = default_colors_dict['blue_sb_dark']
                ):

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    edge_color = [assign_edge_color_from_score(e,ranks) for e in G.edges()]

    x_jittered = np.array([st.t(df=6, scale=x_jit).rvs(1,random_state=np.random.RandomState(seed=seed+x)) for x in G.nodes()])[:, 0]
    positions = {i: (x_jittered[i], ranks[i]) for i in np.arange(A.shape[0])}
    if node_labels is None:
        if nodeId2Label is not None:
            node_labels = {n: nodeId2Label[n] for n in np.arange(A.shape[0])}

    plot_net_hard(G, positions, cm=cm, node_labels=node_labels, ax=ax,
                      ms=ms, plot_arrows=True, edge_color=edge_color,node_size=node_size,
                  plot_labels=plot_labels,u=u,node_color=node_color
                  )
