from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import N_TARGET_NODES,N_SOURCE_NODES
def plot_source (graph):
    hsv_modified = cm.get_cmap('twilight', 256)  # create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.55, 0.88, 100000)))
    plt.figure()
    # trie = np.ma.masked_where(trie == 0, trie)
    newcmp.set_bad(color="#631120")
    plt.pcolormesh(graph, cmap=newcmp)
    plt.ylim(N_SOURCE_NODES, 0)
    plt.colorbar()

    plt.savefig('source.png')

    return plt.show()
def plot_target (graph,predicted):
    hsv_modified = cm.get_cmap('twilight', 256)  # create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.55, 0.88, 100000)))
    plt.figure()
    # trie = np.ma.masked_where(trie == 0, trie)
    newcmp.set_bad(color="#631120")
    plt.pcolormesh(graph, cmap=newcmp)
    plt.ylim(N_TARGET_NODES, 0)
    plt.colorbar()
    if predicted:
        plt.savefig('predicted_target.png')
    else:
        plt.savefig('target.png')

    return plt.show()