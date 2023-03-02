import numpy as np

import mddtw
from cycler import cycler
from numba import njit
from numba.experimental import jitclass
from numba.typed import List

from dtaidistance import dtw_visualisation as dtwvis

# TODO: fix buffer for step sizes
# TODO: deal with diagonal
class MDDTW:

    def __init__(self, series, gamma=1.0, tau=0.0, delta=0.0, delta_factor=0.5, l_min=4, l_max=None, step_sizes=None, use_c=False):
        if step_sizes is None:
            step_sizes = [(1, 1), (1, 0), (0, 1)]
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
        self.delta = delta
        self.delta_factor = delta_factor
        # The LC matrix
        self._cam = None
        # Self similarity matrix
        self._am = None
        # LC paths
        self._paths = None
        # List containing fitness values
        self._fitnesses = None
        self.use_c = use_c

    def align(self):
        if self._am is None:
            self._am  = affinity_matrix_ndim(self.series, self.series, gamma=self.gamma, only_triu=True)
        if self._cam is None:
            # this is not supported here
            if self.use_c:
                self._cam = mddtw.cumulative_affinity_matrix_multidim(self.series, gamma=self.gamma, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor, steps=self.step_sizes)
            else:
                self._cam = cumulative_affinity_matrix(self._am, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor, step_sizes=self.step_sizes, only_triu=True)

    def kbest_paths(self, buffer, k=None):
        if buffer is None:
            buffer = self.l_min // 2

        if self._cam is None:
            self.align()

        # mask region as if diagonal calculated
        mask = np.ones(self._cam.shape, dtype=np.bool)
        mask[np.triu_indices(len(mask), k=buffer)] = False

        # hardcode diagonal
        paths, _ = kbest_paths(self._cam, k, mask, l_min=self.l_min, buffer=buffer, step_sizes=self.step_sizes)
        diagonal = np.repeat([np.arange(len(self.series))], 2, axis=0).T

        # self._paths = List()
        self._paths = []
        self._paths.append(Path(diagonal, np.ones(len(diagonal))))
        
        for path in paths:
            r, c = path[:, 0], path[:, 1]
            affs = self._am[r, c]
            self._paths.append(Path(path, affs))
            # also add mirrored path here
            path_mirrored = np.zeros(path.shape, dtype=np.int32)
            path_mirrored[:, 0] = np.copy(path[:, 1])
            path_mirrored[:, 1] = np.copy(path[:, 0])
            self._paths.append(Path(path_mirrored, affs))

        return self._paths

    def induced_paths(self, s, e, mask=None):
        return _induced_paths(s, e, self.series, self._paths, mask, self.l_min, self.l_max)

    def induced_segments(self, s, e, mask=None):
        induced_paths = self.induced_paths(s, e, mask)
        return row_projections(induced_paths)

    def calculate_fitnesses(self, start_mask, end_mask, mask, allowed_overlap=0, pruning=True):  
        # def _calculate_fitnesses(start_mask, end_mask, mask, n, paths, l_min, l_max, allowed_overlap=0, pruning=True) 
        fitnesses = _calculate_fitnesses(start_mask, end_mask, mask, n=len(self.series), paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap, pruning=pruning)
        return np.array(fitnesses)
    
    def calculate_fitnesses_parallel(self, start_mask, end_mask, mask, allowed_overlap=0, pruning=True, nb_processes=4):
        import multiprocessing as mp
        import functools

        n = len(self.series)
        # _calculate_fitnesses(start_mask, end_mask, mask, n, paths, l_min, l_max, allowed_overlap=0, pruning=True)
        f = functools.partial(_calculate_fitnesses, end_mask=end_mask, mask=mask, n=n, paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap, pruning=pruning)

        # decompose the start mask, each mask should have approximately the same number of zeros
        pool    = mp.Pool(nb_processes)
        results = pool.map(f, [row for row in split_start_mask(n, start_mask, nb_processes)])
        pool.close()
        pool.join()

        # combine the results
        fitnesses = [fitness for result in results for fitness in result]
        if fitnesses:
            fitnesses = np.vstack(fitnesses)
        return np.array(fitnesses)
    
    # iteratively finds the best motif
    def kbest_motif_sets(self, k=None, start_mask=None, end_mask=None, mask=None, allowed_overlap=0, pruning=True):
        n = len(self.series)
        # handle masks
        if start_mask is None:
            start_mask = np.full(n, False)
        if end_mask is None:
            end_mask   = np.full(n, False)
        if mask is None:
            mask       = np.full(n, False)

        dmask = np.full(n, True)
        for path in self._paths[1:]:
            dmask[path[0][1]:path[-1][1]+1] = False
        mask = np.logical_or(mask, dmask)
        
        start_mask[-self.l_min+1:] = True
        end_mask[:self.l_min]      = True

        # iteratively find best motif sets
        motif_sets = []

        while (k is None or len(motif_sets) < k):

            if np.all(mask) or np.all(start_mask) or np.all(end_mask):
                break

            start_mask[mask] = True
            end_mask[mask]   = True
        
            # fitnesses = self.calculate_fitnesses(start_mask, end_mask, mask, overlap=overlap, pruning=pruning)
            fitnesses = self.calculate_fitnesses_parallel(start_mask, end_mask, mask, allowed_overlap=allowed_overlap, pruning=pruning)

            if len(fitnesses) == 0:
                break

            if len(motif_sets) == 0:
                self._fitnesses = fitnesses

            # best motif
            i_best = np.argmax(fitnesses[:, 2])
            best = fitnesses[i_best]

            (s, e) = int(best[0]), int(best[1])
            occs = row_projections(_induced_paths(s, e, self.series, self._paths, mask, self.l_min, self.l_max))
            for (s_o, e_o) in occs:
                l_occ = e_o - s_o
                # if overlap = 0.5, ensure one single t to be masked
                mask[s_o + int(allowed_overlap * l_occ) - 1 : e_o - int(allowed_overlap * l_occ)] = True
            motif_sets.append((best, occs))
            
        return motif_sets

    # requires dtaidistance
    def plot_lc(self):
        max_v = np.amax(self.step_sizes[:, 0])
        max_h = np.amax(self.step_sizes[:, 1])
        # fig, ax = dtwvis.plot_warpingpaths(self.series, self.series, self._cam[max_h - 1:, max_v - 1:], path=-1)
        rgb_cycler = cycler(color=[u'#00407a', u'#2ca02c', u'#c00000'])
        fig, ax = dtwvis.plot_warpingpaths(self.series, self.series, self._cam[max_h - 1:, max_v - 1:], path=-1, cycler=rgb_cycler)
        for p in self._paths:
            dtwvis.plot_warpingpaths_addpath(ax, p.path)
            # dtwvis.plot_warpingpaths_addpath(ax, p.path, style='-')
        return fig, ax

    def plot_lc_full(self):
        max_v = np.amax(self.step_sizes[:, 0])
        max_h = np.amax(self.step_sizes[:, 1])
        cam = self._cam
        cam = cam + cam.T - np.diag(np.diag(cam))
        # fig, ax = dtwvis.plot_warpingpaths(self.series, self.series, cam[max_h - 1:, max_v - 1:], path=-1)
        rgb_cycler = cycler(color=[u'#00407a', u'#2ca02c', u'#c00000'])
        fig, ax = dtwvis.plot_warpingpaths(self.series, self.series, cam[max_h - 1:, max_v - 1:], path=-1, cycler=rgb_cycler)
        for i, p in enumerate(self._paths):
            # dtwvis.plot_warpingpaths_addpath(ax, p.path)
            dtwvis.plot_warpingpaths_addpath(ax, p.path, style='-')
        return fig, ax

    # requires dtaidistance
    def plot_lc_add_induced_paths(self, ax, s, e, mask=None, color='green'):
        induced_paths = self.induced_paths(s, e, mask)
        for p in induced_paths:
            dtwvis.plot_warpingpaths_addpath(ax, p, color=color)
        return ax

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
        
# @jitclass([("path", int32[:, :]), ("sims", float64[:]), ("row_index", int32[:]), ("col_index", int32[:]), ("rs", int32), ("re", int32), ("cs", int32), ("ce", int32)])
class Path:

    def __init__(self, path, sims):
        assert len(path) == len(sims)
        self.path = path
        self.sims = sims.astype(np.float64)
        self.rs = path[0][0]
        self.re = path[len(path) - 1][0] + 1
        self.cs = path[0][1]
        self.ce = path[len(path) - 1][1]  + 1
        self._construct_index(path)

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_index(self, path):
        curr_row = path[0][0]
        curr_col = path[0][1]

        last_row = path[-1][0]
        last_col = path[-1][1]

        row_index = np.zeros(last_row - self.rs + 1, dtype=np.int32)
        col_index = np.zeros(last_col - self.cs + 1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != curr_row:
                row_index[curr_row - self.rs + 1 : path[i][0] - self.rs + 1] = i
                curr_row = path[i][0]

            if path[i][1] != curr_col:
                col_index[curr_col - self.cs + 1 : path[i][1] - self.cs + 1] = i
                curr_col = path[i][1]
        
        self.row_index = row_index
        self.col_index = col_index

    # returns the index of the first occurrence of the given row
    def find_row(self, row):
        assert row - self.rs >= 0 and row - self.rs < len(self.row_index)
        return self.row_index[row - self.rs]

    # returns the index of the first occurrence of the given column
    def find_col(self, col):
        assert col - self.cs >= 0 and col - self.cs < len(self.col_index)
        return self.col_index[col - self.cs]


# @njit(cache=True)
def _induced_paths(s, e, series, paths, mask, l_min, l_max):
    if mask is None:
        mask = np.full(len(series), False)

    induced_paths = []
    for p in paths:
        if p.cs <= s and e <= p.ce:
            pi, pj = p.find_col(s), p.find_col(e-1)
            s_i, e_i = p[pi][0], p[pj][0] + 1
            if not np.any(mask[s_i:e_i]): # and e_i - s_i >= l_min and e_i - s_i <= l_max:
                induced_path = np.copy(p.path[pi:pj+1])
                induced_paths.append(induced_path)

    return induced_paths

# @njit(cache=True, parallel=True)
# @njit(cache=True)
def _calculate_fitnesses(start_mask, end_mask, mask, n, paths, l_min, l_max, allowed_overlap=0, pruning=True):
    ss = np.where(start_mask == False)[0]
    fitnesses = []

    css = np.array([path.cs for path in paths])
    ces = np.array([path.ce for path in paths])

    nbp = len(paths)

    pis = np.zeros(nbp, dtype=np.int32)
    pjs = np.zeros(nbp, dtype=np.int32)
    iss = np.zeros(nbp, dtype=np.int32)
    ies = np.zeros(nbp, dtype=np.int32)

    # 1 means relevant
    pmask = np.full(nbp, False)

    for s in ss:

        # if start_mask[s]:
            # continue

        smask = css <= s

        for e in range(s + l_min, min(n + 1, s + l_max + 1)):
            
            if end_mask[e-1]:
                continue

            if np.any(mask[s:e]):
                break

            emask = ces >= e
            pmask = smask & emask

            # no match
            if not np.any(pmask[1:]):
                break

            ps  = np.flatnonzero(pmask)
            for p in ps:
                path = paths[p]
                pis[p] = pi = path.find_col(s)
                pjs[p] = pj = path.find_col(e-1)
                iss[p] = path[pi][0]
                ies[p] = path[pj][0] + 1
                if np.any(mask[iss[p]:ies[p]]): # or ies[p] - iss[p] < l_min or ies[p] - iss[p] > l_max:
                    pmask[p] = False

            if not np.any(pmask[1:]):
                break

            # from here on only consider unmasked
            ps = np.flatnonzero(pmask)
            # sort iss and ies
            iss_ = iss[pmask]
            ies_ = ies[pmask]

            perm = np.argsort(iss_)
            iss_ = iss_[perm]
            ies_ = ies_[perm]

            skip = False
            overlaps = []
            # check overlap
            for i in range(1, len(iss_)):
                if ies_[i - 1] > iss_[i] + 1:
                    overlap = ies_[i - 1] - (iss_[i] + 1)
                    # if overlap > allowed_overlap:
                    if overlap > allowed_overlap * (ies_[i - 1] - iss_[i - 1]) // 2 or overlap > allowed_overlap * (ies_[i] - iss_[i]) // 2:
                        skip = True
                        break
                    overlaps.append(overlap)

            if skip:
                if pruning:
                    break
                else:
                    continue

            coverage = np.sum(ies_ - iss_) - np.sum(np.array(overlaps))
            n_coverage = (coverage - (e - s)) / float(n)

            score = 0
            total_length = 0
            for p in ps:
                score += np.sum(paths[p].sims[pis[p]:pjs[p]+1])
                total_length += (pjs[p] - pis[p] + 1)

            n_score = (score - (e - s)) / float(total_length)

            fit = 0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            # Calculate the fitness value
            if fit > 0:
                fitnesses.append((s, e, fit, n_coverage, n_score))

    return fitnesses

# project paths to the first axis
# @njit(cache=True)
def row_projections(paths):
    return [(p[0][0], p[len(p)-1][0]+1) for p in paths]

# project paths to the second axis
# @njit(cache=True)
def col_projections(paths):
    return [(p[0][1], p[len(p)-1][1]+1) for p in paths]

@njit(cache=True)
def segment_overlaps(segments):
    overlaps = []
    perm = np.argsort(segments[:, 0])
    sorted_segments = segments[perm, :]
    for i in range(1, len(sorted_segments)):
        if sorted_segments[i - 1][1] > sorted_segments[i][0] + 1:
            overlaps.append(sorted_segments[i - 1][1] - (sorted_segments[i][0] + 1))
    return np.array(overlaps, dtype=np.int32)

@njit(cache=True)
def cumulative_affinity_matrix(am, tau=0.0,  delta=0.0, delta_factor=1.0, step_sizes=np.array([[1, 1], [1, 0], [0, 1]]), window=None, only_triu=False):
    n, m = am.shape

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
            aff     = am[i, j]

            inds    = np.array([i + max_v, j + max_h]) - step_sizes
            max_aff = np.amax(np.array([d[r, c] for (r, c) in inds]))

            if aff < tau:
                d[i + max_v, j + max_h] = max(0, delta + delta_factor * max_aff)
            else:
                d[i + max_v, j + max_h] = max(0, aff + max_aff)
    return d


@njit(cache=True)
def max_warping_path(d, mask, r, c, step_sizes=np.array([[1, 1], [1, 0], [0, 1]])):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    path = []
    while r >= max_v and c >= max_h:

        path.append((r - max_v, c - max_h))

        inds = np.array([r, c]) - step_sizes

        values = np.array([d[i, j]    for (i, j) in inds])
        masked = np.array([mask[i, j] for (i, j) in inds])
        i_max = np.argmax(values)

        if masked[i_max]:
            break

        r -= step_sizes[i_max, 0]
        c -= step_sizes[i_max, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)

@njit(cache=True)
def kbest_paths(d, k, mask, l_min=2, buffer=0, step_sizes=np.array([[1, 1], [1, 0], [0, 1]])):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    rs, cs = np.nonzero(d == 0)
    for i in range(len(rs)):
        mask[rs[i], cs[i]] = True

    maxx, maxy = d.shape

    rs, cs = np.nonzero(d)
    values = np.array([d[rs[i], cs[i]] for i in range(len(rs))])
    
    perm = np.argsort(values)
    rs = rs[perm]
    cs = cs[perm]

    i = len(rs) - 1

    ki = 0
    paths = []

    while (k is None or ki < k) and i >= 0:
        path = None

        while path is None:

            # find the best unmasked
            while (mask[rs[i], cs[i]]):
                i -= 1
                if i < 0:
                    return paths, d

            r, c = rs[i], cs[i]

            if r < max_v or c < max_h:
                return paths, d
            
            path = max_warping_path(d, mask, r, c, step_sizes=step_sizes)
            for (x, y) in path:
                mask[x + max_h, y + max_v] = True

            if (path[-1][0] - path[0][0] + 1) < l_min // 2 or (path[-1][1] - path[0][1] + 1) < l_min // 2:
            # if (path[-1][0] - path[0][0] + 1) < l_min or (path[-1][1] - path[0][1] + 1) < l_min:
                path = None

        # TODO: this can be done more efficiently
        for (x, y) in path:
            xx = x + max_v
            mask[xx, max(0, y + max_h - buffer):min(maxx, y + max_h + buffer + 1)] = True
            yy = y + max_h
            mask[max(0, x + max_v - buffer):min(maxy, x + max_v + buffer + 1), yy] = True

        ki += 1
        paths.append(path)
    return paths, d

@njit(cache=True)
def affinity_matrix_ndim(series1, series2, gamma=1.0, window=None, only_triu=False):
    n, m = len(series1), len(series2)

    if window is None:
        window = max(n, m)

    am = np.zeros((n, m))
    for i in range(n):

        j_start = max(0, i - max(0, n - m) - window + 1)
        if only_triu:
            j_start = max(i, j_start)

        j_end   = min(n, i + max(0, m - n) + window)

        affinities = np.exp(-gamma * np.sum(np.power(series1[i, :] - series2[j_start:j_end, :], 2), axis=1))
        am[i, j_start:j_end] = affinities

    return am

@njit(cache=True)
def to_segments(bit_array):
    diff = np.diff(bit_array)
    ss = np.flatnonzero(diff ==  1) + 1
    es = np.flatnonzero(diff == -1) + 1
    if bit_array[0]:
        ss = np.concatenate((np.zeros(1, dtype=np.int32), ss))
    if bit_array[-1]:
        n = len(bit_array)
        es = np.concatenate((es, n * np.ones(1, dtype=np.int32)))
    return np.vstack((ss, es)).T

# @njit(cache=True)
def split_start_mask(n, start_mask, nb_masks):
    start_mask_matrix = np.full((nb_masks, n), True)
    nb  = np.sum(~start_mask)

    cnt = 0
    i   = 0
    s   = 0

    for e in range(n):
        cnt += int(~start_mask[e])
        if cnt == np.ceil(nb / nb_masks):
            start_mask_matrix[i, s:e] = start_mask[s:e] 
            cnt = 0
            s = e
            i += 1
    if s < n:
        start_mask_matrix[nb_masks-1, s:] = start_mask[s:] 
    return [row for row in start_mask_matrix]

