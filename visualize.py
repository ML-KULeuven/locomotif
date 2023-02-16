import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_motif_sets(series, motifs, gt=None):
    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)

    n = len(series)
    fig, axs = plt.subplots(len(motifs) + 1, 1, figsize=(12, (len(motifs) + 1) * 2), sharex=True, sharey=True)
    axs[0].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
    axs[0].plot(range(len(series)), series, lw=1)
    axs[0].legend([f"dim {d}" for d in range(series.shape[1])])
    axs[0].set_xlim((0, n))

    if gt is not None:
        plot_ground_truth_ax(axs[0], gt, n)

    for i, occs in enumerate(motifs):
        axs[i+1].set_title(f"Motif Set {i}, k: {len(occs)}")
        for s_occ, e_occ in occs:
            axs[i+1].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
            axs[i+1].plot(range(s_occ, e_occ), series[s_occ : e_occ, :], alpha=1, lw=1)
            axs[i+1].axvline(x=s_occ, c='k', linestyle=':', lw=0.25)
            axs[i+1].axvline(x=e_occ, c='k', linestyle=':', lw=0.25)

    plt.tight_layout()
    return fig, axs

def plot_ground_truth(series, gt):
    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)
    n = len(series)
    fig, ax = plt.subplots(figsize=(12, 2), sharex=True, sharey=True)
    ax.set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
    ax.plot(range(len(series)), series, lw=1)
    ax.legend([f"dim {d}" for d in range(series.shape[1])])
    ax.set_xlim((0, n))
    plot_ground_truth_ax(ax, gt, n)
    return fig, ax
    

def plot_ground_truth_ax(ax, gt, n):
    for key in gt.keys():
        for (s, e) in gt[key]:
            ax.axvline(x=s, c='k', linestyle=':', lw=0.25)
            ax.axvline(x=e, c='k', linestyle=':', lw=0.25)

            text_x = (s + ((e - s) // 2)) / float(n)
            text_y = 0.90
            ax.text(text_x, text_y, str(key), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    return ax
