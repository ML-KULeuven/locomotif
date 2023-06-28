import numpy as np
import matplotlib.pyplot as plt
import os

import locomotif.locomotif as locomotif
import locomotif.visualize as visualize


def main():
    # A time series is represented as a (n x d)-numpy array where n is the length of the time series and d is its dimensionality
    path_to_series = os.path.join(".", "datasets", "ecg-heartbeat-av.csv")
    f = open(path_to_series)
    series = np.array(f.readlines(), dtype=np.double)

    print(series.shape)

    # z-normalise time series
    series = (series - np.mean(series, axis=0)) / np.std(series, axis=0)

    # Parameter rho determines the 'strictness' of the algorithm  
    #   - higher -> more strict (time series subsequences match less easily)
    #   - lower  -> less strict (time series subsequences match more easily) 
    rho = 0.55

    # Number of motifs to be found
    nb_motifs = 2

    fs = 128  # sampling frequency

    # Heartbeats last 0.6s - 1s (equivalent to 60-100 bpm)
    l_min = int(0.6 * fs)
    l_max = int(  1 * fs)

    # This parameter determines how much the motifs may overlap (intra and inter motif set)
    overlap = 0.25

    # start_mask specifies whether a motif can start at a certain time point (same for end_mask)
    # for example: motifs can only start in the second half of the time series:
    n = len(series)
    start_mask = np.full(n, True)
    start_mask[:n//2] = False

    motif_sets = locomotif.apply_locomotif(series, rho, l_min, l_max, nb_motifs, overlap=overlap, start_mask=start_mask, end_mask=None)
    print(motif_sets)
    fig, ax = visualize.plot_motif_sets(series, motif_sets)
    plt.savefig('example.png')
    plt.show()

if __name__ == "__main__":
   main()