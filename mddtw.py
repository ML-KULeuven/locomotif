import numpy as np

import mddtw

from numba import njit
from numba.experimental import jitclass
from numba.typed import List

from numba import int32, float64, bool_

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
            # this is not supported
            if self.use_c:
                self._cam = mddtw.cumulative_affinity_matrix_multidim(self.series, gamma=self.gamma, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor, steps=self.step_sizes)
            else:
                self._cam = cumulative_affinity_matrix(self._am, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor, step_sizes=self.step_sizes, only_triu=True)

    def kbest_paths(self, buffer, k=None):
        if buffer is None:
            buffer = self.l_min // 2

        if self._cam is None:
            self.align()
        paths, _ = kbest_paths(self._cam, k, l_min=self.l_min, buffer=buffer, step_sizes=self.step_sizes)

        self._paths = List()
        for path in paths:
            r, c = path[:, 0], path[:, 1]
            self._paths.append(Path(path, self._am[r, c]))
        return self._paths

    def _calculate_fitnesses(self, allowed_overlap=0, pruning=True, mask=None):
        if mask is None:
            mask = np.zeros(len(self.series), dtype=np.bool)
        fitnesses = _calculate_fitnesses(self.series, mask=mask, paths=self._paths, l_min=self.l_min, l_max=self.l_max, allowed_overlap=allowed_overlap, pruning=pruning)
        return np.array(fitnesses)

    def induced_segments(self, s, e, mask=None):
        if mask is None:
            mask = np.zeros(len(self.series), dtype=np.bool)

        segments = []
        for i, p in enumerate(self._paths):

            if p.cs <= s and e <= p.ce:
                s_i, e_i = p[p.find_col(s)][0], p[p.find_col(e-1)][0] + 1
                if not np.any(mask[s_i:e_i]):
                    segments.append((s_i, e_i))

            if i == 0:
                continue

            if p.rs <= s and e <= p.re:
                s_i, e_i = p[p.find_row(s)][1], p[p.find_row(e-1)][1] + 1
                if not np.any(mask[s_i:e_i]):
                    segments.append((s_i, e_i))

        return segments

    # iteratively finds the best motif
    def kbest_motifs(self, k=1, overlap=0, pruning=True):
        n = len(self.series)
        mask = np.zeros(n, dtype=np.bool)
        motifs = []
        ki = k
        for k in range(ki):
            
            fitnesses = self._calculate_fitnesses(mask=mask, allowed_overlap=overlap, pruning=pruning)

            if len(fitnesses) == 0:
                return motifs

            if k == 0:
                self._fitnesses = fitnesses
            
            # best motif
            (s, e, _, _, _) = fitnesses[np.argmax(fitnesses[:, 2])].astype(int)

            occs = self.induced_segments(s, e, mask)
            for (s_o, e_o) in occs:
                mask[s_o + overlap:e_o - overlap] = True

            motifs.append(((s, e), occs))

        return motifs

    # subsequences that are farthest away from other subsequences (alternatively: highest distances to closest neighbors) (or just no matches ...)
    def discords(self):
        discords = _discords(self.series, self._paths, self.l_min, self.l_max)
        return discords


@njit
def _discords(series, paths, l_min, l_max):
    mask = np.zeros(len(series), dtype=np.int32)
    for path in paths[1:]:
        mask[path[0][1]:path[-1][1]+1] = 1
        mask[path[0][0]:path[-1][0]+1] = 1
    segments = to_segments(1 - mask)
    segments = [(s, e) for (s, e) in segments if l_min < (e - s) and (e - s) < l_max]
    return segments
        

@njit
def _calculate_fitnesses(series, mask, paths, l_min, l_max, allowed_overlap=0, pruning=True):
    n = len(series)

    # TODO: do not do this everytime
    discords = _discords(series, paths, l_min, l_max)
    # bitarray containing where a motif can start
    ss = np.ones(n, dtype=np.int32)
    ss[-l_min+1:] = 0
    for (s_d, e_d) in discords:
        ss[s_d:e_d] = 0
    ss[mask] = 0
    ss = np.flatnonzero(ss)

    # mirror the paths: 
    all_paths = [paths[0]]
    for path in paths[1:]:
        path_mirrored = np.zeros(path.path.shape, dtype=np.int32)
        path_mirrored[:, 0] = np.copy(path.path[:, 1])
        path_mirrored[:, 1] = np.copy(path.path[:, 0])
        all_paths.append(path)
        all_paths.append(Path(path_mirrored, path.sims))
        
    fitnesses = []

    css = np.array([path.cs for path in all_paths])
    ces = np.array([path.ce for path in all_paths])

    nbp = len(all_paths)

    pis = np.zeros(nbp, dtype=np.int32)
    pjs = np.zeros(nbp, dtype=np.int32)
    iss = np.zeros(nbp, dtype=np.int32)
    ies = np.zeros(nbp, dtype=np.int32)

    # 1 means relevant
    pmask = np.zeros(nbp, dtype=bool_)

    for s in ss:

        smask = css <= s
        for e in range(s + l_min, min(n + 1, s + l_max + 1)):
            
            if np.any(mask[s:e]):
                break

            emask = ces >= e
            pmask = smask & emask

            # no match
            if not np.any(pmask[1:]):
                break

            ps  = np.flatnonzero(pmask)
            # from here on only consider unmasked
            for p in ps:
                path = all_paths[p]
                pis[p] = pi = path.find_col(s)
                pjs[p] = pj = path.find_col(e-1)
                iss[p] = path[pi][0]
                ies[p] = path[pj][0] + 1
                if np.any(mask[iss[p]:ies[p]]):
                    pmask[p] = False


            if not np.any(pmask[1:]):
                break

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
                    if overlap > allowed_overlap:
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
                score += np.sum(all_paths[p].sims[pis[p]:pjs[p]+1])
                total_length += (pjs[p] - pis[p] + 1)


            n_score = (score - (e - s)) / float(total_length)

            fit = 0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            # Calculate the fitness value
            if fit > 0:
                fitnesses.append((s, e, fit, n_coverage, n_score))

    return fitnesses


@jitclass([("path", int32[:, :]), ("sims", float64[:]), ("row_index", int32[:]), ("col_index", int32[:]), ("rs", int32), ("re", int32), ("cs", int32), ("ce", int32)])
class Path:

    def __init__(self, path, sims):
        assert len(path) == len(sims)
        self.path = path
        self.sims = sims.astype(float64)
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


# project paths to the first axis
@njit
def row_projections(paths):
    return [(p[0][0], p[-1][0]+1) for p in paths]

# project paths to the second axis
@njit
def col_projections(paths):
    return [(p[0][1], p[-1][1]+1) for p in paths]

@njit
def segment_overlaps(segments):
    overlaps = []
    perm = np.argsort(segments[:, 0])
    sorted_segments = segments[perm, :]
    for i in range(1, len(sorted_segments)):
        if sorted_segments[i - 1][1] > sorted_segments[i][0] + 1:
            overlaps.append(sorted_segments[i - 1][1] - (sorted_segments[i][0] + 1))
    return np.array(overlaps, dtype=np.int32)

@njit
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


@njit
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

@njit
def kbest_paths(d, k, l_min=2, buffer=0, step_sizes=np.array([[1, 1], [1, 0], [0, 1]])):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    mask = np.zeros(d.shape, dtype=np.bool_)

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

            if path[-1][0] - path[0][0] + 1 < l_min or path[-1][1] - path[0][1] + 1 < l_min:
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

@njit
def affinity_matrix(series1, series2, gamma=1.0, window=None, only_triu=False):
    n, m = len(series1), len(series2)

    if window is None:
        window = max(n, m)

    am = np.full((n, m),  -np.inf)
    for i in range(n):

        j_start = max(0, i - max(0, n - m) - window + 1)
        if only_triu:
            j_start = max(i, j_start)

        j_end   = min(n, i + max(0, m - n) + window)

        affinities = np.exp(-gamma * np.power(series1[i] - series2[j_start:j_end], 2))
        am[i, j_start:j_end] = affinities

    return am

@njit
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

@njit
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


def estimate_tau_from_std(series, f, gamma=None):
    diffm = np.std(series, axis=0)
    diffp = f * diffm

    if gamma is None:
        gamma = 1 / np.dot(diffp, diffp)

    tau = np.exp(- gamma * np.dot(diffp, diffp))
    return tau, gamma