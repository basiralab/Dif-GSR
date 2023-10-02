import yaml
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def plot_heatmap(data, title, save_path):
    arr = data
    hsv_modified = cm.get_cmap('twilight', 256)  # create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.55, 0.88, 100000)))
    plt.figure()
    # trie = np.ma.masked_where(trie == 0, trie)
    newcmp.set_bad(color="#631120")
    plt.pcolormesh(arr, cmap=newcmp)
    plt.ylim(arr.shape[1], 0)
    plt.colorbar()
    plt.savefig(f"{save_path}/{title}.png")





