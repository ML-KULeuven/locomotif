import matplotlib.pyplot as plt
import numpy as np

def plot_motif_sets(series, motifs):
    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)
    fig, axs = plt.subplots(len(motifs) + 1, 1, figsize=(12, (len(motifs) + 1) * 2), sharex=True, sharey=True)
    axs[0].plot(range(len(series)), series, lw=1)
    axs[0].legend([f"dim {d}" for d in range(series.shape[1])])
    for i, (_, occs) in enumerate(motifs):
        axs[i+1].set_title(f"Motif Set {i}, k: {len(occs)}")
        for s_occ, e_occ in occs:
            axs[i+1].set_prop_cycle(None)
            axs[i+1].plot(range(s_occ, e_occ), series[s_occ : e_occ, :], alpha=1, lw=1)
            axs[i+1].axvline(x=s_occ, c='k', linestyle=':', lw=0.25)
            axs[i+1].axvline(x=e_occ, c='k', linestyle=':', lw=0.25)
    plt.tight_layout()
    return fig, axs
