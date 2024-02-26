import numpy as np

from numba import int32, float64
from numba import njit
from numba.typed import List
from numba.experimental import jitclass

def apply_locomotif(series, rho, l_min, l_max, nb=None, start_mask=None, end_mask=None, overlap=0.5, warping=True):
    if start_mask is None:
        start_mask = np.full(len(series), True)
    if end_mask is None:
        end_mask   = np.full(len(series), True)

    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)
        
    gamma = 1
    sm  = similarity_matrix_ndim(series, series, gamma, only_triu=True)
    tau = estimate_tau_from_am(sm, rho)

    delta_a = -2*tau
    delta_m = 0.5
    step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])

    locomotif = LoCoMotif(series=series, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, l_min=l_min, l_max=l_max, step_sizes=step_sizes)
    locomotif._sm = sm
    locomotif.align()
    locomotif.kbest_paths(vwidth=l_min // 2)
    motif_sets = []
    for (_, motif_set), _ in locomotif.kbest_motif_sets(nb=nb, allowed_overlap=overlap, start_mask=start_mask, end_mask=end_mask, pruning=False):
        motif_sets.append(motif_set)
    return motif_sets


class LoCoMotif:

    def __init__(self, series, gamma=1.0, tau=0.5, delta_a=0.5, delta_m=0.5, l_min=5, l_max=None, step_sizes=None):
        if step_sizes is None:
            step_sizes = [(1, 1), (2, 1), (1, 2)]
        if l_max is None:
            l_max = len(series)

        if series.ndim == 1:
            series = np.expand_dims(series, axis=1)
            
        self.series = np.array(series, dtype=np.float64)

        self.l_min = l_min
        self.l_max = l_max
        self.step_sizes = step_sizes
        # LC args
        self.gamma = gamma
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m
        # Cumulative similiarity matrix
        self._csm = None
        # Self similarity matrix
        self._sm = None
        # LC Paths
        self._paths = None

    def align(self):
        if self._sm is None:
            self._sm  = similarity_matrix_ndim(self.series, self.series, gamma=self.gamma, only_triu=True)
        if self._csm is None:
            self._csm = cumulative_similarity_matrix(self._sm, tau=self.tau, delta_a=self.delta_a, delta_m=self.delta_m, step_sizes=self.step_sizes, only_triu=True)

    def kbest_paths(self, vwidth, nbp=None):
        if vwidth is None:
            vwidth = max(10, self.l_min // 2)
        vwidth = max(10, vwidth)
            
        if self._csm is None:
            self.align()

        # mask region as if diagonal is already found as a path
        mask = np.full(self._csm.shape, True)
        mask[np.triu_indices(len(mask), k=vwidth)] = False

        paths = _kbest_paths(self._csm, nbp, mask, l_min=self.l_min, vwidth=vwidth, step_sizes=self.step_sizes)

        # hardcode diagonal (needed if step_sizes = [(1, 1), (0, 1), (1, 0)])
        diagonal = np.vstack(np.diag_indices(len(self.series))).astype(np.int32).T
        self._paths = List()
        # self._paths = []
        self._paths.append(Path(diagonal, np.ones(len(diagonal))))
        
        for path in paths:
            i, j = path[:, 0], path[:, 1]
            path_similarities = self._sm[i, j]
            self._paths.append(Path(path, path_similarities))
            # also add mirrored path here
            path_mirrored = np.zeros(path.shape, dtype=np.int32)
            path_mirrored[:, 0] = np.copy(path[:, 1])
            path_mirrored[:, 1] = np.copy(path[:, 0])
            self._paths.append(Path(path_mirrored, path_similarities))

        return self._paths

    def induced_paths(self, b, e, mask=None):
        return _induced_paths(b, e, self.series, self._paths, mask)

    def calculate_fitnesses(self, start_mask, end_mask, mask, allowed_overlap=0, pruning=True):  
        fitnesses = _calculate_fitnesses(start_mask, end_mask, mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap, pruning=pruning)
        return np.array(fitnesses)
    

    # iteratively finds the best motif
    def kbest_motif_sets(self, nb=None, start_mask=None, end_mask=None, mask=None, allowed_overlap=0, pruning=False):
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
        
            fitnesses = self.calculate_fitnesses(start_mask, end_mask, mask, allowed_overlap=allowed_overlap, pruning=pruning)
            # fitnesses = self.calculate_fitnesses_parallel(start_mask, end_mask, mask, allowed_overlap=allowed_overlap, pruning=pruning)

            if len(fitnesses) == 0:
                break

            # best candidate
            i_best = np.argmax(fitnesses[:, 2])
            best = fitnesses[i_best]

            candidate = (b, e) = int(best[0]), int(best[1])
            motif_set = vertical_projections(_induced_paths(b, e, self.series, self._paths, mask))
            for (bm, em) in motif_set:
                l = em - bm
                mask[bm + int(allowed_overlap * l) - 1 : em - int(allowed_overlap * l)] = True
            motif_set.insert(0, motif_set.pop(motif_set.index(candidates)))

            current_nb += 1
            yield (best, motif_set), fitnesses
            
    def get_paths(self):
        return [path.path for path in self._paths]
            

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
        
@jitclass([("path", int32[:, :]), ("sim", float64[:]), ("cumsim", float64[:]), ("index_i", int32[:]), ("index_j", int32[:]), ("i1", int32), ("il", int32), ("j1", int32), ("jl", int32)])
class Path:

    def __init__(self, path, sim):
        assert len(path) == len(sim)
        self.path = path
        self.sim = sim.astype(np.float64)
        self.cumsim = np.concatenate((np.array([0.0]), np.cumsum(sim)))
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

    
# project paths to the vertical axis
def vertical_projections(paths):
    return [(p[0][0], p[len(p)-1][0]+1) for p in paths]

# project paths to the horizontal axis
def horizontal_projections(paths):
    return [(p[0][1], p[len(p)-1][1]+1) for p in paths]


@njit(cache=True)
def _kbest_paths(d, nbp, mask, l_min=2, vwidth=10, step_sizes=np.array([[1, 1], [2, 1], [1, 2]])):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    is_, js_ = np.nonzero(d <= 0)
    for index_best in range(len(is_)):
        mask[is_[index_best], js_[index_best]] = True

    is_, js_ = np.nonzero(d)
    values = np.array([d[is_[i], js_[i]] for i in range(len(is_))])
    
    perm = np.argsort(values)
    is_ = is_[perm]
    js_ = js_[perm]

    index_best = len(is_) - 1

    curr_nbp = 0
    paths = []

    while (nbp is None or curr_nbp < nbp) and index_best >= 0:
        path = None

        while path is None:

            # Find the best unmasked
            while (mask[is_[index_best], js_[index_best]]):
                index_best -= 1
                if index_best < 0:
                    return paths

            i_best, j_best = is_[index_best], js_[index_best]
            if i_best < max_v or j_best < max_h:
                return paths
            
            path = max_warping_path(d, mask, i_best, j_best, step_sizes=step_sizes)
            for (x, y) in path:
                mask[x + max_h, y + max_v] = True

            if (path[-1][0] - path[0][0] + 1) < l_min and (path[-1][1] - path[0][1] + 1) < l_min:
                path = None

        # Buffer: Bresenham's algorithm
        (xc, yc) = path[0] + np.array((max_v, max_h))
        for (xt, yt) in path[1:] + np.array((max_v, max_h)):
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
        
        # Add path
        curr_nbp += 1
        paths.append(path)
    return paths

# @njit(cache=True)
def _induced_paths(b, e, series, paths, mask):
    if mask is None:
        mask = np.full(len(series), False)

    induced_paths = []
    for p in paths:
        if p.j1 <= b and e <= p.jl:
            kb, ke = p.find_j(b), p.find_j(e-1)
            bm, em = p[kb][0], p[ke][0] + 1
            if not np.any(mask[bm:em]):
                induced_path = np.copy(p.path[kb:ke+1])
                induced_paths.append(induced_path)

    return induced_paths

# @njit(cache=True, parallel=True)
@njit(cache=True)
def _calculate_fitnesses(start_mask, end_mask, mask, paths, l_min, l_max, allowed_overlap=0, pruning=True):
    start_indices = np.where(start_mask == True)[0]
    fitnesses = []

    n = len(start_mask)

    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nbp = len(paths)

    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)
    bs = np.zeros(nbp, dtype=np.int32)
    es = np.zeros(nbp, dtype=np.int32)

    for b in start_indices:

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

            ps  = np.flatnonzero(pmask)
            for p in ps:
                path = paths[p]
                kbs[p] = pi = path.find_j(b)
                kes[p] = pj = path.find_j(e-1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                if np.any(mask[bs[p]:es[p]]): # or ies[p] - iss[p] < l_min or ies[p] - iss[p] > l_max:
                    pmask[p] = False

            if not np.any(pmask[1:]):
                break

            # sort iss and ies
            bs_ = bs[pmask]
            es_ = es[pmask]

            perm = np.argsort(bs_)
            bs_ = bs_[perm]
            es_ = es_[perm]

            # overlaps   
            len_     = es_ - bs_
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps  = np.maximum(es_[:-1] - bs_[1:] - 1, 0)
            
            if np.any(overlaps > allowed_overlap * len_[:-1]): 
                if pruning:
                    break
                else:
                    continue

            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = (coverage - (e - b)) / float(n)

            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p]+1] - paths[p].cumsim[kbs[p]]

            n_score = (score - (e - b)) / float(np.sum(kes[pmask] - kbs[pmask] + 1))

            fit = 0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            # Calculate the fitness value
            if fit > 0:
                fitnesses.append((b, e, fit, n_coverage, n_score))

    return fitnesses


@njit(cache=True)
def similarity_matrix_ndim(series1, series2, gamma=1.0, window=None, only_triu=False):
    n, m = len(series1), len(series2)

    if window is None:
        window = max(n, m)

    sm = np.full((n, m), -np.inf)
    for i in range(n):

        j_start = max(0, i - max(0, n - m) - window + 1)
        if only_triu:
            j_start = max(i, j_start)

        j_end   = min(m, i + max(0, m - n) + window)

        similarities = np.exp(-gamma * np.sum(np.power(series1[i, :] - series2[j_start:j_end, :], 2), axis=1))
        sm[i, j_start:j_end] = similarities

    return sm

@njit(cache=True)
def cumulative_similarity_matrix(sm, tau=0.0, delta_a=0.0, delta_m=1.0, step_sizes=np.array([[1, 1], [2, 1], [1, 2]]), window=None, only_triu=False):
    n, m = sm.shape

    if window is None:
        window = max(n, m)

    max_v = np.amax(step_sizes[:, 0])
    max_h = np.amax(step_sizes[:, 1])

    d = np.zeros((n + max_v, m + max_h))

    for i in range(n):
        
        j_start = max(0, i - max(0, n - m) - window + 1)
        if only_triu:
            j_start = max(i, j_start)
        j_end   = min(m, i + max(0, m - n) + window)
        
        for j in range(j_start, j_end):
            sim     = sm[i, j]

            indices    = np.array([i + max_v, j + max_h]) - step_sizes
            max_cumsim = np.amax(np.array([d[i_, j_] for (i_, j_) in indices]))

            if sim < tau:
                d[i + max_v, j + max_h] = max(0, delta_a + delta_m * max_cumsim)
            else:
                d[i + max_v, j + max_h] = max(0, sim + max_cumsim)
    return d


@njit(cache=True)
def max_warping_path(d, mask, i, j, step_sizes=np.array([[1, 1], [2, 1], [1, 2]])):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    path = []
    while i >= max_v and j >= max_h:

        path.append((i - max_v, j - max_h))

        indices = np.array([i, j]) - step_sizes

        values = np.array([d[i_, j_]    for (i_, j_) in indices])
        masked = np.array([mask[i_, j_] for (i_, j_) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        i -= step_sizes[argmax, 0]
        j -= step_sizes[argmax, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)

# @njit(cache=True)
# def split_start_mask(n, start_mask, nb_masks):
#     start_mask_matrix = np.full((nb_masks, n), False)
#     nb  = np.sum(start_mask)

#     cnt = 0
#     i   = 0
#     s   = 0

#     for e in range(n):
#         cnt += int(start_mask[e])
#         if cnt == np.ceil(nb / nb_masks):
#             start_mask_matrix[i, s:e] = start_mask[s:e] 
#             cnt = 0
#             s = e
#             i += 1
#     if s < n:
#         start_mask_matrix[nb_masks-1, s:] = start_mask[s:] 
#     return [row for row in start_mask_matrix]

    
# def calculate_fitnesses_parallel(self, start_mask, end_mask, mask, allowed_overlap=0, pruning=True, nb_processes=4):
#     import multiprocessing as mp
#     import functools

#     n = len(self.series)
#     # _calculate_fitnesses(start_mask, end_mask, mask, paths, l_min, l_max, allowed_overlap=0, pruning=True)
#     f = functools.partial(_calculate_fitnesses, end_mask=end_mask, mask=mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap, pruning=pruning)

#     # decompose the start mask, each mask should have approximately the same number of zeros
#     pool    = mp.Pool(nb_processes)
#     results = pool.map(f, [row for row in split_start_mask(n, start_mask, nb_processes)])
#     pool.close()
#     pool.join()

#     # combine the results
#     fitnesses = [fitness for result in results for fitness in result]
#     if fitnesses:
#         fitnesses = np.vstack(fitnesses)
#     return np.array(fitnesses)