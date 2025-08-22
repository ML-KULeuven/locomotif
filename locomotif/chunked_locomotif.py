
import numpy as np
from .path import project_to_vertical_axis, Path
from .locomotif import _mask_motif_set, _induced_paths

import multiprocessing

from abc import ABC, abstractmethod
import tqdm

class ChunkedLoCoMotif(ABC):

    @abstractmethod
    def __init__(self, T, l_min, l_max, parallel=False, n_processes=1):        
        l_min = max(4, l_min)
        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)

        self._parallel   = parallel
        self._n_processes = n_processes
        if not parallel:
            self._n_processes = 1

        self.P = None
        # Chunked LoCo instance
        self._loco = None
        
    @abstractmethod
    def find_best_paths(self):        
        self.P = self._loco.find_best_paths(l_min=self.l_min, vwidth=self.l_min // 2)
        return self.P

    @abstractmethod
    def find_best_motif_sets(self, nb=None, overlap=0.25):
        if self.P is None:
            self.find_best_paths()
            
        chunk_begin_cols = self._loco._chunk_begin_cols
        chunk_end_cols   = self._loco._chunk_end_cols
        chunk_begin_rows = self._loco._chunk_begin_rows
        chunk_end_rows   = self._loco._chunk_end_rows

        T = self._loco.T
        n = np.int32(len(T))
        Nc = len(chunk_begin_cols)

        # Group paths by column, and add the row offset
        P = {c: [] for c in range(Nc)}
        for (r, c) in self.P.keys():
            Tr, Tc = T[chunk_begin_rows[r]:chunk_end_rows[r]], T[chunk_begin_cols[c]:chunk_end_cols[c]]
            for path in self.P[r, c]:
                # Recalculate similarities on the path
                similarities = np.exp(np.sum(-self._loco.gamma.T * np.power(Tr[path[:, 0], :] - Tc[path[:, 1], :], 2), axis=1))
                # Offset the row indices
                positions    = np.vstack((path[:, 0] + chunk_begin_rows[r], path[:, 1])).T
                P[c].append(Path(positions.astype(np.int32), similarities.astype(np.float32)))
                
        P_concat = {}
        for c in P.keys():
            P_pos = np.vstack([path.path for path in P[c]])
            P_sim = np.concatenate([path.similarities for path in P[c]])
            P_len = np.array([len(path) for path in P[c]], dtype=np.int32)
            P_concat[c] = (P_pos, P_sim, P_len)
                
        global_mask = np.full(n, False)
        start_mask  = np.full(n, True)
        end_mask    = np.full(n, True)

        print(f"Finding motif sets: Using {self._n_processes} processes.")
        ## Create the pool
        pool = None
        if self._parallel:
            nprocesses = self._n_processes
            pool = multiprocessing.Pool(processes=nprocesses)
        
        motif_sets = []

        try:
            b_repr = np.zeros(Nc, dtype=np.int32)
            e_repr = np.zeros(Nc, dtype=np.int32)
            fitnesses = np.zeros(Nc)
            
            common_args = {'n': n, 'l_min': self.l_min, 'l_max': self.l_max, 'nu': overlap}
            
            while (nb is None or len(motif_sets) < nb):

                args = {}
                
                for c in range(Nc):
                    ## Note: candidates do not have to be checked in one of the two overlapping chuncks. Can be implemented through start_mask. But we ignore for now.
                    col_begin_index = chunk_begin_cols[c]
                    col_end_index   = chunk_end_cols[c]
                    col_mask   = global_mask[col_begin_index:col_end_index]
                    start_mask = start_mask[col_begin_index:col_end_index]
                    end_mask   = end_mask[col_begin_index:col_end_index]
                    P_pos, P_sim, P_len = P_concat[c]
                    args[c] = {'P_pos': P_pos, 'P_sim': P_sim, 'P_len': P_len} | common_args | {'row_mask': global_mask, 'col_mask': col_mask, 'start_mask': start_mask, 'end_mask': end_mask, 'keep_fitnesses': False}


                if self._parallel:
                    results = list(tqdm.tqdm(
                            pool.imap(_find_best_candidate_wrapper, [list(arg.values()) for arg in args.values()]),
                            total=len(args),
                            desc="Processing chunks"
                        ))
                else:
                    results = [_find_best_candidate_wrapper(arg.values()) for arg in tqdm.tqdm(args.values(), total=len(args), desc="Processing chunks")]

                for c in range(Nc):
                    (b_repr[c], e_repr[c]), fitnesses[c], _ = results[c]
                    
                c_best = np.argmax(fitnesses)
                if fitnesses[c_best] == 0.0:
                    break

                motif_set = [project_to_vertical_axis(induced_path) for induced_path in _induced_paths(b_repr[c_best], e_repr[c_best], global_mask, P[c_best])]
                global_mask = _mask_motif_set(global_mask, motif_set, overlap=overlap)
                
                b_best, e_best = b_repr[c_best] + chunk_begin_cols[c_best], e_repr[c_best] + chunk_begin_cols[c_best]
                motif_sets.append(((b_best, e_best), motif_set))

        finally:

            if pool is not None:
                pool.close()
                pool.join()

        return motif_sets


def _find_best_candidate_wrapper(args):
    P_pos, P_sim, P_len, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses = args
    return _find_best_candidate_parallelizable(P_pos, P_sim, P_len, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses=keep_fitnesses)

import numba
from numba import int32, float64, float32, boolean
from numba import njit
from numba.types import Tuple, UniTuple
from .locomotif import _find_best_candidate # _find_best_candidate(P, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses=False)
@njit(Tuple((UniTuple(int32, 2), float32, float32[:, :]))(int32[:, :], float32[:], int32[:], int32, int32, int32, float64, boolean[:], boolean[:], boolean[:], boolean[:], boolean))
def _find_best_candidate_parallelizable(P_pos, P_sim, P_len, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses=False):
    path_bounds = np.concatenate((np.array([0]), np.cumsum(P_len)))
    P = numba.typed.List()
    for i in range(len(P_len)):
        path_pos = P_pos[path_bounds[i]:path_bounds[i+1], :]
        path_sim = P_sim[path_bounds[i]:path_bounds[i+1]]
        path = Path(path_pos, path_sim)
        P.append(path)
    return _find_best_candidate(P, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses=keep_fitnesses)