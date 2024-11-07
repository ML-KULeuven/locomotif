import numpy as np
from . import util

import numba
from numba import int32, float64, float32, boolean
from numba import njit
from numba.typed import List
from numba.experimental import jitclass


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
    lcm.align()
    lcm.find_best_paths(vwidth=l_min // 2)
    # Find the `nb` best motif sets
    motif_sets = []
    for representative, motif_set, _ in lcm.find_best_motif_sets(nb=nb, overlap=overlap, start_mask=start_mask, end_mask=end_mask):
        motif_sets.append((representative, motif_set))
    return motif_sets


def get_locomotif_instance(ts, l_min, l_max, rho=None, warping=True, ts2=None):
    return LoCoMotif.instance_from_rho(ts, l_min=l_min, l_max=l_max, rho=rho, warping=warping, ts2=ts2)


class LoCoMotif:

    def __init__(self, ts, l_min, l_max, gamma=1.0, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=None, ts2=None):
        if step_sizes is None:
            step_sizes = np.array([(1, 1), (2, 1), (1, 2)])

        self._sm_symmetric = True  # Is the SSM symmetric (true when comparing to itself)
        if ts.ndim == 1:
            ts = np.expand_dims(ts, axis=1)
        self.ts = np.array(ts, dtype=np.float32)
        if ts2 is None:
            self.ts2 = self.ts
        else:
            if ts2.ndim == 1:
                ts2 = np.expand_dims(ts2, axis=1)
            self.ts2 = np.array(ts2, dtype=np.float32)
            self._sm_symmetric = False

        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)
        self.step_sizes = step_sizes.astype(np.int32)
        # LoCo arguments
        self.gamma = gamma
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m
        # Self-similarity matrix
        self._sm = None
        # Cumulative similiarity matrix
        self._csm = None
        # Local warping paths
        self._paths = None

    @classmethod
    def instance_from_rho(cls, ts, l_min, l_max, rho=None, warping=True, ts2=None):
        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5
        # Make ts of shape (n,) of shape (n, 1) such that it can be handled as a multivariate ts
        if ts.ndim == 1:
            ts = np.expand_dims(ts, axis=1)
        ts = np.array(ts, dtype=np.float32)
        if ts2 is None:
            ts2 = ts
            issym = True
        else:
            if ts2.ndim == 1:
                ts2 = np.expand_dims(ts2, axis=1)
            ts2 = np.array(ts2, dtype=np.float32)
            issym = False
        # Check whether the time series is z-normalized. If not, give a warning.
        if not util.is_unitstd(ts): # util.is_znormalized(ts):
            import warnings
            warnings.warn(
                "It is highly recommended to z-normalize the input time series so that it has a standard deviation of 1 before applying LoCoMotif to it.")

        gamma = 1
        # Determine values for tau, delta_a, delta_m based on the ssm and rho
        sm = similarity_matrix_ndim(ts, ts2, gamma, only_triu=issym)
        tau = estimate_tau_from_am(sm, rho)

        delta_a = 2 * tau
        delta_m = 0.5
        # Determine step_sizes based on warping
        step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])
        lcm = cls(ts=ts, l_min=l_min, l_max=l_max, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m,
                  step_sizes=step_sizes, ts2=ts2)
        lcm._sm = sm
        return lcm

    def align(self):
        if self._sm is None:
            self._sm  = similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma,
                                               only_triu=self._sm_symmetric)
        if self._csm is None:
            self._csm = cumulative_similarity_matrix(self._sm, tau=self.tau,
                                                     delta_a=self.delta_a, delta_m=self.delta_m,
                                                     step_sizes=self.step_sizes,
                                                     only_triu=self._sm_symmetric)

    
    def find_best_paths(self, vwidth):
        if vwidth is None:
            vwidth = max(10, self.l_min // 2)
        vwidth = max(10, vwidth)
            
        if self._csm is None:
            self.align()

        # Hardcode diagonal (this is only needed if step_sizes=[(1, 1), (0, 1), (1, 0)]).
        #   - First, mask region around the diagional as if the diagonal is already found as a path.
        #   - After applying LoCo, add the diagonal to the result.
        mask = np.full(self._csm.shape, self._sm_symmetric)
        if self._sm_symmetric:
            mask[np.triu_indices(len(mask), k=vwidth)] = False
        # LoCo is only applied to the part of the SSM above the diagonal. Later, the mirrored versions of the found paths are added.
        paths = _find_best_paths(self._csm, mask, l_min=self.l_min, vwidth=vwidth, step_sizes=self.step_sizes)
        self._paths = List()
        if self._sm_symmetric:
            diagonal = np.vstack(np.diag_indices(len(self.ts))).astype(np.int32).T
            self._paths.append(Path(diagonal, np.ones(len(diagonal)).astype(np.float32)))
        
        for path in paths:
            i, j = path[:, 0], path[:, 1]
            path_similarities = self._sm[i, j]
            self._paths.append(Path(path, path_similarities))
            if self._sm_symmetric:
                # Also add the mirrored path
                path_mirrored = np.zeros(path.shape, dtype=np.int32)
                path_mirrored[:, 0] = np.copy(path[:, 1])
                path_mirrored[:, 1] = np.copy(path[:, 0])
                self._paths.append(Path(path_mirrored, path_similarities))

        return self._paths

    def induced_paths(self, b, e, mask=None):
        if mask is None:
            mask = np.full(len(self.ts), False)

        induced_paths = []
        for p in self._paths:
            if p.j1 <= b and e <= p.jl:
                kb, ke = p.find_j(b), p.find_j(e-1)
                bm, em = p[kb][0], p[ke][0] + 1
                if not np.any(mask[bm:em]):
                    induced_path = np.copy(p.path[kb:ke+1])
                    induced_paths.append(induced_path)

        return induced_paths

    # iteratively finds the best motif set
    def find_best_motif_sets(self, nb=None, start_mask=None, end_mask=None, overlap=0.0):
        n = len(self.ts)
        # handle masks
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
            # fitnesses = self.calculate_fitnesses_parallel(start_mask, end_mask, mask, allowed_overlap=allowed_overlap)

            if best_fitness == 0.0:
                break

            motif_set = vertical_projections(self.induced_paths(b, e, mask))
            for (bm, em) in motif_set:
                l = em - bm
                mask[bm + int(overlap * l) - 1 : em - int(overlap * l)] = True

            current_nb += 1
            yield (b, e), motif_set, fitnesses
            
    def get_paths(self):
        return [path.path for path in self._paths]
    
    def get_ssm(self):
        return self._sm


@jitclass([("path", int32[:, :]), ("sim", float32[:]), ("cumsim", float32[:]), ("index_i", int32[:]), ("index_j", int32[:]), ("i1", int32), ("il", int32), ("j1", int32), ("jl", int32)])
class Path:

    def __init__(self, path, sim):
        assert len(path) == len(sim)
        self.path = path
        self.sim = sim.astype(np.float32)
        self.cumsim = np.concatenate((np.array([0.0], dtype=np.float32), np.cumsum(sim)))
        self.i1 = path[0][0]
        self.il = path[len(path) - 1][0] + 1
        self.j1 = path[0][1]
        self.jl = path[len(path) - 1][1] + 1
        self._construct_index(path)

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_index(self, path):
        i_curr = path[0][0]
        j_curr = path[0][1]

        index_i = np.zeros(self.il - self.i1, dtype=np.int32)
        index_j = np.zeros(self.jl - self.j1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != i_curr:
                index_i[i_curr - self.i1 + 1 : path[i][0] - self.i1 + 1] = i
                i_curr = path[i][0]

            if path[i][1] != j_curr:
                index_j[j_curr - self.j1 + 1 : path[i][1] - self.j1 + 1] = i
                j_curr = path[i][1]
        
        self.index_i = index_i
        self.index_j = index_j

    # returns the index of the first occurrence of the given row
    def find_i(self, i):
        assert i - self.i1 >= 0 and i - self.i1 < len(self.index_i)
        return self.index_i[i - self.i1]

    # returns the index of the first occurrence of the given column
    def find_j(self, j):
        assert j - self.j1 >= 0 and j - self.j1 < len(self.index_j)
        return self.index_j[j - self.j1]

    
def estimate_tau_from_std(ts, f, gamma=None):
    diffm = np.std(ts, axis=0)
    diffp = f * diffm
    if gamma is None:
        gamma = 1 / np.dot(diffp, diffp)
    tau = np.exp(- gamma * np.dot(diffp, diffp))
    return tau, gamma

# page 194 of Fundamentals of Music Processing
def estimate_tau_from_am(am, rho):
    tau = np.quantile(am[np.triu_indices(len(am))], rho, axis=None)
    return tau
        
# Project paths to the vertical axis
def vertical_projections(paths):
    return [(p[0][0], p[len(p)-1][0]+1) for p in paths]

# Project paths to the horizontal axis
def horizontal_projections(paths):
    return [(p[0][1], p[len(p)-1][1]+1) for p in paths]


@njit(float32[:, :](float32[:, :], float32[:, :], float64, boolean))
def similarity_matrix_ndim(series1, series2, gamma=1.0, only_triu=False):
    n, m = len(series1), len(series2)

    sm = np.full((n, m), -np.inf, dtype=float32)
    for i in range(n):

        j_start = i if only_triu else 0
        j_end   = m
        
        similarities = np.exp(-gamma * np.sum(np.power(series1[i, :] - series2[j_start:j_end, :], 2), axis=1))
        sm[i, j_start:j_end] = similarities

    return sm

@njit(float32[:, :](float32[:, :], float64, float64, float64, int32[:, :], boolean))
def cumulative_similarity_matrix(sm, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=np.array([[1, 1], [2, 1], [1, 2]]), only_triu=False):
    n, m = sm.shape
    max_v = np.amax(step_sizes[:, 0])
    max_h = np.amax(step_sizes[:, 1])

    d = np.zeros((n + max_v, m + max_h), dtype=float32)

    for i in range(n):
        
        j_start = i if only_triu else 0
        j_end   = m
        
        for j in range(j_start, j_end):
            sim     = sm[i, j]

            indices    = np.array([i + max_v, j + max_h]) - step_sizes
            max_cumsim = np.amax(np.array([d[i_, j_] for (i_, j_) in indices]))

            if sim < tau:
                d[i + max_v, j + max_h] = max(0, delta_m * max_cumsim - delta_a)
            else:
                d[i + max_v, j + max_h] = max(0, sim + max_cumsim)
    return d

@njit(int32[:, :](float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def max_warping_path(d, mask, i, j, step_sizes=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    path = []
    while i >= max_v and j >= max_h:

        path.append((i - max_v, j - max_h))

        indices = np.array([i, j], dtype=np.int32) - step_sizes

        values = np.array([d[i_, j_]    for (i_, j_) in indices])
        masked = np.array([mask[i_, j_] for (i_, j_) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        i, j = i - step_sizes[argmax, 0], j - step_sizes[argmax, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)

@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32))
def mask_path(path, mask, max_v, max_h):
    for (x, y) in path:
        mask[x + max_h, y + max_v] = True
    return mask

@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32, int32))
def mask_vicinity(path, mask, max_v, max_h, vwidth=10):
    (xc, yc) = path[0] + np.array((max_v, max_h))
    for (xt, yt) in path[1:] + np.array([max_v, max_h]):
        dx  =  xt - xc
        dy  =  yc - yt
        err = dx + dy
        while xc != xt or yc != yt:
            mask[xc-vwidth:xc+vwidth+1, yc] = True
            mask[xc, yc-vwidth:yc+vwidth+1] = True
            e = 2 * err
            if e > dy:
                err += dy
                xc  += 1
            if e < dx:
                err += dx
                yc  += 1
    mask[xt-vwidth:xt+vwidth+1, yt] = True
    mask[xt, yt-vwidth:yt+vwidth+1] = True
    return mask

@njit(numba.types.List(int32[:, :])(float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def _find_best_paths(d, mask, l_min=2, vwidth=10, step_sizes=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    # Mask all values <= 0
    is_, js_ = np.nonzero(d <= 0)
    for index_best in range(len(is_)):
        mask[is_[index_best], js_[index_best]] = True

    # Sort indices based on values in D
    is_, js_ = np.nonzero(d)
    values = np.array([d[is_[i], js_[i]] for i in range(len(is_))])
    perm = np.argsort(values)
    is_ = is_[perm]
    js_ = js_[perm]

    index_best = len(is_) - 1
    paths = []
    
    # Keep constructing paths until all positions have been processed
    while index_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)

        # Keep iterating until valid path is constructed
        path_found = False
        while not path_found:

            # Find the best unmasked index in D
            while (mask[is_[index_best], js_[index_best]]):
                index_best -= 1
                if index_best < 0:
                    return paths

            i_best, j_best = is_[index_best], js_[index_best]

            if i_best < max_v or j_best < max_h:
                return paths
            
            # Reconstruct the path
            path = max_warping_path(d, mask, i_best, j_best, step_sizes=step_sizes)
            # Mask the indices on path
            mask = mask_path(path, mask, max_v, max_h)

            # Discard if any projection of path is shorter than the minimum motif length  
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        # Path goes through: use Bresenham's algorithm to mask its vicinity
        mask = mask_vicinity(path, mask, max_v, max_h, vwidth)
        
        # Add the path to the result
        paths.append(path)

    return paths


@njit(numba.types.Tuple((numba.types.UniTuple(int32, 2), float32, float32[:, :]))(boolean[:], boolean[:], boolean[:], numba.types.ListType(Path.class_type.instance_type), int32, int32, float64, boolean))
def _find_best_candidate(start_mask, end_mask, mask, paths, l_min, l_max, overlap=0.0, keep_fitnesses=False):
    fitnesses = []    
    n = len(start_mask)

    # j1s and jls respectively contain the column index of the first and last position of all paths
    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nbp = len(paths)

    # bs and es will respectively contain the start and end indices of the motifs in the  candidate motif set of the current candidate [b : e].
    bs  = np.zeros(nbp, dtype=np.int32)
    es  = np.zeros(nbp, dtype=np.int32)

    # kbs and kes will respectively contain the index on the path (\in [0 : len(path)]) where the path crosses the vertical line through b and e.
    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)

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

            # If there are not paths that cross both the vertical line through b and e, skip the candidate.
            if not np.any(pmask[1:]):
                break

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_j(b)
                kes[p] = pj = path.find_j(e-1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                # Check overlap with previously found motifs.
                if np.any(mask[bs[p]:es[p]]): # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[p] = False

            # If the candidate only matches with itself, skip it.
            if not np.any(pmask[1:]):
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
            overlaps  = np.maximum(es_[:-1] - bs_[1:] - 1, 0)
            
            # Overlap check within motif set
            if np.any(overlaps > overlap * len_[:-1]): 
                continue

            # Calculate normalized coverage
            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = (coverage - (e - b)) / float(n)

            # Calculate normalized score
            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p]+1] - paths[p].cumsim[kbs[p]]
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
    
    fitnesses = np.array(fitnesses, dtype=np.float32) if keep_fitnesses else np.empty((0, 5), dtype=np.float32)
    return best_candidate, best_fitness, fitnesses


