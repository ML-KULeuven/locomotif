import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIG_SIZE = 14


def plot_motif_sets(series, motif_sets, gt=None):
    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)

    n = len(series)
    fig, axs = plt.subplots(len(motif_sets) + 1, 1, figsize=(12, (len(motif_sets) + 1) * 2), sharex=True, sharey=True)

    if np.array(axs).ndim == 0:
        axs = [axs]

    axs[0].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
    axs[0].plot(range(len(series)), series, lw=1.5)
    axs[0].legend([f"dim {d+1}" for d in range(series.shape[1])], fontsize=SMALL_SIZE)
    axs[0].set_xlim((0, n))

    if gt is not None:
        plot_ground_truth_ax(axs[0], gt, n)

    for i, motif_set in enumerate(motif_sets):
        axs[i+1].set_title(f"Motif Set {i+1}, k: {len(motif_set)}", fontsize=BIG_SIZE)
        for s_m, e_m in motif_set:
            axs[i+1].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
            axs[i+1].plot(range(s_m, e_m), series[s_m : e_m, :], alpha=1, lw=1.5)
            axs[i+1].axvline(x=s_m, c='k', linestyle=':', lw=0.25)
            axs[i+1].axvline(x=e_m, c='k', linestyle=':', lw=0.25)

    plt.tight_layout()
    return fig, axs

def plot_ground_truth_ax(ax, gt, n):
    for key in gt.keys():
        for (s, e) in gt[key]:
            ax.axvline(x=s, c='k', linestyle=':', lw=0.25)
            ax.axvline(x=e, c='k', linestyle=':', lw=0.25)
            
            text_x = (s + ((e - s) // 2)) / float(n)
            text_y = 0.90
            ax.text(text_x, text_y, str(key), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    return ax

