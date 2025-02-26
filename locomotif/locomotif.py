import numpy as np

from . import loco
from .path import Path, project_to_vertical_axis

import numba
from numba import int32, float64, float32, boolean
from numba import njit
from numba.typed import List



def apply_locomotif(ts, l_min, l_max, rho=None, nb=None, start_mask=None, end_mask=None, overlap=0.0, warping=True):
    """Apply the LoCoMotif algorithm to find motif sets in the given time ts.

    :param ts: Univariate or multivariate time series, with the time axis being the 0-th dimension.
    :param l_min: Minimum length of the representative motifs.
    :param l_max: Maximum length of the representative motifs.
    :param rho: The strictness parameter between 0 and 1. It is the quantile of the similarity matrix to use as the threshold for the LoCo algorithm.
    :param nb: Maximum number of motif sets to find.
    :param start_mask: Mask for the starting time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param end_mask: Mask for the ending time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param overlap: Maximum allowed overlap between motifs, between 0 and 0.5. A new motif β can be discovered only when |β ∩ β'|/|β'| is less than this value for all existing motifs β'.
    :param warping: Whether warping is allowed (True) or not (False).
    
    :return: motif_sets: a list of motif sets, where each motif set is a list of segments as tuples.
    """   
    # Get a locomotif instance
    lcm = get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=warping)
    # Apply LoCo
    lcm.find_best_paths(vwidth=l_min // 2)
    # Find the `nb` best motif sets
    motif_sets = []
    for representative, motif_set, _ in lcm.find_best_motif_sets(nb=nb, overlap=overlap, start_mask=start_mask, end_mask=end_mask):
        motif_sets.append((representative, motif_set))
    return motif_sets

def get_locomotif_instance(ts, l_min, l_max, rho=None, warping=True):
    return LoCoMotif.instance_from_rho(ts, l_min=l_min, l_max=l_max, rho=rho, warping=warping)


class LoCoMotif:

    def __init__(self, ts, l_min, l_max, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=None):
        if step_sizes is None:
            step_sizes = np.array([(1, 1), (2, 1), (1, 2)])
            
        self.ts = ts
        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)
        # LoCo instance
        self._loco = loco.LoCo(ts, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, step_sizes=step_sizes)

    @classmethod
    def instance_from_rho(cls, ts, l_min, l_max, rho=None, warping=True):
        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5  
        step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])
        lcm = cls(ts=ts, l_min=l_min, l_max=l_max)
        lcm._loco = loco.LoCo.instance_from_rho(ts, rho, gamma=None, step_sizes=step_sizes)
        return lcm

    def find_best_paths(self, vwidth):
        vwidth = np.maximum(10, self.l_min // 2)
        paths = self._loco.find_best_paths(self.l_min, vwidth)
       
        # LoCo finds paths the part of the SSM above the diagonal. 
        # Here, the paths are converted to Path objects, and the mirrored versions of the found paths are added.
        self._paths = List()
        
        for path in paths:
            i, j = path[:, 0], path[:, 1]
            path_similarities = self.self_similarity_matrix[i, j]
            self._paths.append(Path(path, path_similarities))
            # Also add the mirrored path
            # Do not mirror the diagonal
            if np.all(i == j):
                continue
            path_mirrored = np.zeros(path.shape, dtype=np.int32)
            path_mirrored[:, 0] = j
            path_mirrored[:, 1] = i
            self._paths.append(Path(path_mirrored, path_similarities))

        return self._paths

    def induced_paths(self, b, e, mask=None):
        if mask is None:
            mask = np.full(len(self.ts), False)

        induced_paths = []
        for path in self._paths:
            if b < path.j1 or path.jl < e:
                continue
            induced_path = path.get_subpath_between_col_indices(b, e-1)
            b_m, e_m = project_to_vertical_axis(induced_path)
            if not np.any(mask[b_m:e_m]):
                induced_paths.append(induced_path)

        return induced_paths

    # iteratively finds the best motif set
    def find_best_motif_sets(self, nb=None, start_mask=None, end_mask=None, overlap=0.0):
        n = len(self.ts)
        # Handle masks
        if start_mask is None:
            start_mask = np.full(n, True)
        if end_mask is None:
            end_mask   = np.full(n, True)
    
        assert 0.0 <= overlap and overlap <= 0.5
        assert start_mask.shape == (n,)
        assert end_mask.shape   == (n,)

        # iteratively find best motif sets
        current_nb = 0
        mask       = np.full(n, False)
        while (nb is None or current_nb < nb):

            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask]   = False
        
            (b, e), best_fitness, fitnesses  = _find_best_candidate(start_mask, end_mask, mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, overlap=overlap, keep_fitnesses=False)

            if best_fitness == 0.0:
                break

            motif_set = [project_to_vertical_axis(induced_path) for induced_path in self.induced_paths(b, e, mask)]
            for (b_m, e_m) in motif_set:
                l = e_m - b_m
                l_mask = max(1, int((1 - 2*overlap) * l)) # mask length must be lower bounded by 1 (otherwise, nothing is masked when overlap=0.5)
                mask[b_m + (l - l_mask)//2 : b_m + (l - l_mask)//2 + l_mask] = True

            current_nb += 1
            yield (b, e), motif_set, fitnesses
            
    @property
    def local_warping_paths(self):
        return self._paths
    
    @property
    def self_similarity_matrix(self):
        return self._loco.similarity_matrix
    
    @property
    def cumulative_similarity_matrix(self):
        return self._loco.cumulative_similarity_matrix
    

@njit(numba.types.Tuple((numba.types.UniTuple(int32, 2), float32, float32[:, :]))(boolean[:], boolean[:], boolean[:], numba.types.ListType(Path.class_type.instance_type), int32, int32, float64, boolean))
def _find_best_candidate(start_mask, end_mask, mask, paths, l_min, l_max, overlap=0.0, keep_fitnesses=False):
    fitnesses = []    
    n = len(start_mask)

    # j1s and jls respectively contain the column index of the first and last position of each path
    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nb_paths = len(paths)

    # bs and es will respectively contain the start and end indices of the motifs in the candidate motif set of the current candidate [b : e].
    bs  = np.zeros(nb_paths, dtype=np.int32)
    es  = np.zeros(nb_paths, dtype=np.int32)

    # kbs and kes will respectively contain the index on the path (\in [0 : len(path)]) where the path crosses the vertical line through b and e.
    kbs = np.zeros(nb_paths, dtype=np.int32)
    kes = np.zeros(nb_paths, dtype=np.int32)

    best_fitness   = 0.0
    best_candidate = (0, n) 

    for b in range(n - l_min + 1):
        
        if not start_mask[b]:
            continue
            
        smask = j1s <= b

        for e in range(b + l_min, min(n + 1, b + l_max + 1)):
            
            if not end_mask[e-1]:
                continue

            if np.any(mask[b:e]):
                break

            emask = jls >= e
            pmask = smask & emask

            # If there's only one relevant path, skip the candidate.
            if np.sum(pmask) < 2:
                break

            for path_index in np.flatnonzero(pmask):
                path = paths[path_index]
                kbs[path_index] = kb = path.find_j(b)
                kes[path_index] = ke = path.find_j(e-1)
                bs[path_index] = path[kb][0]
                es[path_index] = path[ke][0] + 1
                # Check overlap with previously found motifs.
                if np.any(mask[bs[path_index]:es[path_index]]): # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[path_index] = False

            # If there's less than matches, skip the candidate.
            if np.sum(pmask) < 2:
                break

            # Sort bs and es on bs such that overlaps can be calculated efficiently
            bs_ = bs[pmask]
            es_ = es[pmask]

            perm = np.argsort(bs_)
            bs_ = bs_[perm]
            es_ = es_[perm]

            # Calculate the overlaps   
            len_     = es_ - bs_
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps  = np.maximum(es_[:-1] - bs_[1:], 0)
            
            # Overlap check within motif set
            if np.any(overlaps > overlap * len_[:-1]): 
                continue

            # Calculate normalized coverage
            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = (coverage - (e - b)) / float(n)

            # Calculate normalized score
            score = 0
            for path_index in np.flatnonzero(pmask):
                score += paths[path_index].cumulative_similarities[kes[path_index]+1] - paths[path_index].cumulative_similarities[kbs[path_index]]

            n_score = (score - (e - b)) / float(np.sum(kes[pmask] - kbs[pmask] + 1))
            
            # Calculate the fitness value
            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)
    
            if fit == 0.0:
                continue

            # Update best fitness
            if fit > best_fitness:
                best_candidate = (b, e)
                best_fitness   = fit

            # Store fitness if necessary
            if keep_fitnesses:
                fitnesses.append((b, e, fit, n_coverage, n_score))
    
    fitnesses = np.array(fitnesses, dtype=np.float32) if keep_fitnesses and fitnesses else np.empty((0, 5), dtype=np.float32)
    return best_candidate, best_fitness, fitnesses


