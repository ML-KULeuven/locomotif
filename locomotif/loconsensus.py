import numpy as np

from .chunked_locomotif import ChunkedLoCoMotif
from .chunked_loco import ChunkedLoCo
from .loco import ensure_multivariate

# Currently, LoConsenus assumes that every comparison between two time series fits into memory, which limits the allowed length of the time series.
# TODO: resolve this by allowing ChunkedLoCo instances to find the paths
class LoConsensus(ChunkedLoCoMotif):

    def __init__(
            self, 
            Ts, l_min, l_max, 
            gamma=1.0, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, equal_weight_dims=False,
            chunk_memory_limit=(1024)**3, # in bytes, default is 1GB 
            parallel=False, n_processes=1
        ):


        Ts = [ensure_multivariate(T) for T in Ts]
        assert np.all([Ts[0].shape[1] == T.shape[1] for T in Ts])
        lengths = [len(T) for T in Ts]
        cumulative_lengths = np.cumsum(np.concatenate(([0], lengths)))
        self.Ts = Ts

        T = np.concatenate(Ts) # TODO: Best to z-normalize before concatenating
        super().__init__(T, l_min, l_max, parallel=parallel, n_processes=n_processes)
        

        self._loco = ChunkedLoCo( 
            T,
            gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, warping=warping, equal_weight_dims=equal_weight_dims,
            parallel=self._parallel, n_processes=self._n_processes, 
            chunk_memory_limit=chunk_memory_limit, 
            chunk_begin_rows=cumulative_lengths[:-1], chunk_end_rows=cumulative_lengths[1:],
            chunk_begin_cols=cumulative_lengths[:-1], chunk_end_cols=cumulative_lengths[1:],
        )
        
    def find_best_paths(self):
        super().find_best_paths()
        Nc = len(self._loco._chunk_begin_cols)
        Nr = len(self._loco._chunk_begin_rows)
        
        for r in range(Nr):
            for c in range(r, Nc):
                mirrored_paths = []
                for path in self.P[r, c]:
                    i, j = path[:, 0], path[:, 1]
                    # If self-comparison, do not mirror the diagonal
                    if r == c and np.all(i == j):
                        continue
                    path_mirrored = np.zeros(path.shape, dtype=np.int32)
                    path_mirrored[:, 0], path_mirrored[:, 1] = j, i
                    mirrored_paths.append(path_mirrored)
                if r == c: 
                    self.P[c, r].extend(mirrored_paths)
                else:
                    self.P[c, r] = mirrored_paths
        return self.P
    
    def find_best_motif_sets(self, nb=None, overlap=0.25):
        motif_sets = super().find_best_motif_sets(nb, overlap)
        
        consensus_motif_sets = []    
        for motif_set in motif_sets:
            (_, motifs) = motif_set
            # motifs grouped by T in Ts
            grouped_motifs = {r: [] for r in range(len(self.Ts))}
            
            for (b, e) in motifs:
                r = np.sum(self._loco._chunk_begin_rows <= b) - 1
                grouped_motifs[r].append((b - self._loco._chunk_begin_rows[r], e - self._loco._chunk_begin_rows[r]))
                
            consensus_motif_sets.append(grouped_motifs)

        return consensus_motif_sets