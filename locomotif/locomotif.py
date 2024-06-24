import numpy as np

import numba
from numba import int32, float64, float32, boolean
from numba import njit
from numba.typed import List
from numba.experimental import jitclass


def apply_locomotif(series, rho, l_min, l_max, nb=None, start_mask=None, end_mask=None, overlap=0.5, warping=True):
    """Apply the LoCoMotif algorithm to find motif sets in the given time series.

    :param series: Univariate or multivariate time series, with the time axis being the 0-th dimension.
    :param rho: The strictness parameter between 0 and 1. It is the quantile of the similarity matrix to use as the threshold for the LoCo algorithm.
    :param l_min: Minimum length of the representative motifs.
    :param l_max: Maximum length of the representative motifs.
    :param nb_motifs: Maximum number of motif sets to find.
    :param start_mask: Mask for the starting time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param end_mask: Mask for the ending time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param overlap: Maximum allowed overlap between motifs, between 0 and 0.5. A new motif β can be discovered only when |β ∩ β'|/|β'| is less than this value for all existing motifs β'.
    :param warping: Whether warping is allowed (True) or not (False).
    
    :return: motif_sets: a list of motif sets, where each motif set is a list of segments as tuples.
    """   
    lcm = get_locomotif_instance(series, rho, l_min, l_max, nb=nb, start_mask=start_mask, end_mask=end_mask, overlap=overlap, warping=warping)
    lcm.align()
    lcm.kbest_paths(vwidth=l_min // 2)
    motif_sets = []
    for (_, motif_set), _ in lcm.kbest_motif_sets(nb=nb, allowed_overlap=overlap, start_mask=start_mask, end_mask=end_mask):
        motif_sets.append(motif_set)
    return motif_sets

def get_locomotif_instance(series, rho, l_min, l_max, nb=None, start_mask=None, end_mask=None, overlap=0.5, warping=True):
    if start_mask is None:
        start_mask = np.full(len(series), True)
    if end_mask is None:
        end_mask   = np.full(len(series), True)

    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)
    
    # TODO
    n = len(series)
    assert len(start_mask) == n
    assert len(end_mask)   == n

    gamma = 1
    series = np.array(series, dtype=np.float32)
    sm  = similarity_matrix_ndim(series, series, gamma, only_triu=True)
    tau = estimate_tau_from_am(sm, rho)

    delta_a = -2*tau
    delta_m = 0.5
    step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])

    lcm = LoCoMotif(series=series, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, l_min=l_min, l_max=l_max, step_sizes=step_sizes)
    lcm._sm = sm
    return lcm

class LoCoMotif:

    def __init__(self, series, gamma=1.0, tau=0.5, delta_a=0.5, delta_m=0.5, l_min=5, l_max=None, step_sizes=None):
        if step_sizes is None:
            step_sizes = np.array([(1, 1), (2, 1), (1, 2)])
        if l_max is None:
            l_max = len(series)

        if series.ndim == 1:
            series = np.expand_dims(series, axis=1)
            
        self.series = np.array(series, dtype=np.float32)

        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)
        self.step_sizes = step_sizes.astype(np.int32)
        # LC args
        self.gamma = gamma
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m
        # Cumulative similiarity matrix
        self._csm = None
        # Self similarity matrix
        self._sm = None
        # Local Warping paths
        self._paths = None

    def align(self):
        if self._sm is None:
            self._sm  = similarity_matrix_ndim(self.series, self.series, gamma=self.gamma, only_triu=True)
        if self._csm is None:
            self._csm = cumulative_similarity_matrix(self._sm, tau=self.tau, delta_a=self.delta_a, delta_m=self.delta_m, step_sizes=self.step_sizes, only_triu=True)

    def kbest_paths(self, vwidth):
        if vwidth is None:
            vwidth = max(10, self.l_min // 2)
        vwidth = max(10, vwidth)
            
        if self._csm is None:
            self.align()

        # Mask region around the diagional as if the diagonal is already found as a path
        mask = np.full(self._csm.shape, True)
        mask[np.triu_indices(len(mask), k=vwidth)] = False

        paths = _kbest_paths(self._csm, mask, l_min=self.l_min, vwidth=vwidth, step_sizes=self.step_sizes)

        # Hardcode diagonal as a path (only needed if step_sizes = [(1, 1), (0, 1), (1, 0)])
        diagonal = np.vstack(np.diag_indices(len(self.series))).astype(np.int32).T
        self._paths = List()
        # self._paths = []
        self._paths.append(Path(diagonal, np.ones(len(diagonal)).astype(np.float32)))
        
        for path in paths:
            i, j = path[:, 0], path[:, 1]
            path_similarities = self._sm[i, j]
            self._paths.append(Path(path, path_similarities))
            # Add mirrored path
            path_mirrored = np.zeros(path.shape, dtype=np.int32)
            path_mirrored[:, 0] = np.copy(path[:, 1])
            path_mirrored[:, 1] = np.copy(path[:, 0])
            self._paths.append(Path(path_mirrored, path_similarities))

        return self._paths

    def induced_paths(self, b, e, mask=None):
        if mask is None:
            mask = np.full(len(series), False)

        induced_paths = []
        for p in self._paths:
            if p.j1 <= b and e <= p.jl:
                kb, ke = p.find_j(b), p.find_j(e-1)
                bm, em = p[kb][0], p[ke][0] + 1
                if not np.any(mask[bm:em]):
                    induced_path = np.copy(p.path[kb:ke+1])
                    induced_paths.append(induced_path)

        return induced_paths

    def calculate_fitnesses(self, start_mask, end_mask, mask, allowed_overlap=0):  
        fitnesses = _calculate_fitnesses(start_mask, end_mask, mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap)
        return fitnesses

    # iteratively finds the best motif set
    def kbest_motif_sets(self, nb=None, start_mask=None, end_mask=None, mask=None, allowed_overlap=0.0):
        n = len(self.series)
        # handle masks
        if start_mask is None:
            start_mask = np.full(n, True)
        if end_mask is None:
            end_mask   = np.full(n, True)
        if mask is None:
            mask       = np.full(n, False)

        # iteratively find best motif sets
        current_nb = 0
        while (nb is None or current_nb < nb):

            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask]   = False
        
            fitnesses = self.calculate_fitnesses(start_mask, end_mask, mask, allowed_overlap=allowed_overlap)
            # fitnesses = self.calculate_fitnesses_parallel(start_mask, end_mask, mask, allowed_overlap=allowed_overlap)

            if len(fitnesses) == 0:
                break

            # best candidate
            i_best = np.argmax(fitnesses[:, 2])
            best = fitnesses[i_best]

            candidate = (b, e) = int(best[0]), int(best[1])
            motif_set = vertical_projections(self.induced_paths(b, e, mask))
            for (bm, em) in motif_set:
                l = em - bm
                mask[bm + int(allowed_overlap * l) - 1 : em - int(allowed_overlap * l)] = True
            motif_set.insert(0, motif_set.pop(motif_set.index(candidate)))

            current_nb += 1
            yield (best, motif_set), fitnesses
            
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

    
def estimate_tau_from_std(series, f, gamma=None):
    diffm = np.std(series, axis=0)
    diffp = f * diffm
    if gamma is None:
        gamma = 1 / np.dot(diffp, diffp)
    tau = np.exp(- gamma * np.dot(diffp, diffp))
    return tau, gamma

# page 194 of Fundamentals of Music Processing
def estimate_tau_from_am(am, rho):
    tau = np.quantile(am[np.triu_indices(len(am))], rho, axis=None)
    return tau
        
# project paths to the vertical axis
def vertical_projections(paths):
    return [(p[0][0], p[len(p)-1][0]+1) for p in paths]

# project paths to the horizontal axis
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
def cumulative_similarity_matrix(sm, tau=0.0, delta_a=0.0, delta_m=1.0, step_sizes=np.array([[1, 1], [2, 1], [1, 2]]), only_triu=False):
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
                d[i + max_v, j + max_h] = max(0, delta_a + delta_m * max_cumsim)
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
def _kbest_paths(d, mask, l_min=2, vwidth=10, step_sizes=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)):
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
        while path.size == 0:

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
            if (path[-1][0] - path[0][0] + 1) < l_min and (path[-1][1] - path[0][1] + 1) < l_min:
                path = np.empty((0, 0), dtype=np.int32)

        # Path goes through: use Bresenham's algorithm to mask its vicinity
        mask = mask_vicinity(path, mask, max_v, max_h, vwidth)
        
        # Add the path to the result
        paths.append(path)

    return paths


@njit(float32[:, :](boolean[:], boolean[:], boolean[:], numba.types.ListType(Path.class_type.instance_type), int32, int32, float64))
def _calculate_fitnesses(start_mask, end_mask, mask, paths, l_min, l_max, allowed_overlap=0.0):
    fitnesses = []    
    n = len(start_mask)

    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nbp = len(paths)

    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)
    bs  = np.zeros(nbp, dtype=np.int32)
    es  = np.zeros(nbp, dtype=np.int32)

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

            # no match
            if not np.any(pmask[1:]):
                break

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_j(b)
                kes[p] = pj = path.find_j(e-1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                if np.any(mask[bs[p]:es[p]]): # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[p] = False

            if not np.any(pmask[1:]):
                break

            # sort bs and es on bs
            bs_ = bs[pmask]
            es_ = es[pmask]

            perm = np.argsort(bs_)
            bs_ = bs_[perm]
            es_ = es_[perm]

            # Calculate overlaps   
            len_     = es_ - bs_
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps  = np.maximum(es_[:-1] - bs_[1:] - 1, 0)
            
            # Overlap check within motif set
            if np.any(overlaps > allowed_overlap * len_[:-1]): 
                continue

            # Calculate the fitness value
            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = (coverage - (e - b)) / float(n)

            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p]+1] - paths[p].cumsim[kbs[p]]

            n_score = (score - (e - b)) / float(np.sum(kes[pmask] - kbs[pmask] + 1))
            
            fit = 0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            if fit > 0:
                fitnesses.append((b, e, fit, n_coverage, n_score))    
    
    if fitnesses:
        return np.array(fitnesses, dtype=np.float32)
    else:
        return np.empty((0, 5), dtype=np.float32)


