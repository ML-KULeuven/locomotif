import numpy as np
import numba
from numba import int32, float64, float32, boolean
from numba import njit

class LoCo:

    def __init__(self, ts, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=None, ts2=None, equal_weight_dims=False):
        if step_sizes is None:
            step_sizes = np.array([(1, 1), (2, 1), (1, 2)])
        
        # If ts2 is specified, we assume it is different from ts. Alternative is self._symmetric = np.array_equal(self.ts, self.ts2) 
        self._symmetric = False
        if ts2 is None:
            self._symmetric = True
            ts2 = ts

        self.ts = np.array(ts, dtype=np.float32)
        self.ts2 = np.array(ts2, dtype=np.float32)

        # Make ts of shape (n,) of shape (n, 1) such that it can be handled as a multivariate ts
        if self.ts.ndim == 1:
            self.ts = np.expand_dims(self.ts, axis=1)
        if self.ts2.ndim == 1:
            self.ts2 = np.expand_dims(self.ts2, axis=1)

        # Handle the gamma argument.
        _, D = self.ts.shape
        if gamma is None:
            # If no value is specified, determine the gamma value(s) based on the input TS.
            if self._symmetric:
                if D == 1 or not equal_weight_dims:
                    gamma = D * [1 / np.std(ts, axis=None)**2]
                else:
                    gamma = [1 / np.std(ts[:, d])**2 for d in range(D)]
            else:
                gamma = D * [1.0]
        # If a single value is specified for gamma, that value is used for every dimension. 
        elif np.isscalar(gamma):
            gamma = D * [gamma]
        # Else, len(gamma) should be equal to the number of dimensions
        else:
            assert np.ndim(gamma) == 1 and len(gamma) == D

        self.gamma = np.array(gamma, dtype=np.float64)
                        
        # LoCo args
        self.step_sizes = step_sizes.astype(np.int32)
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m
        # Self-similarity matrix
        self._sm = None
        # Cumulative similiarity matrix
        self._csm = None
        # Local warping paths
        self._paths = None


    @property
    def similarity_matrix(self):
        return self._sm

    @property
    def cumulative_similarity_matrix(self):
        if self._csm is None:
            return None
        max_v = np.amax(self.step_sizes[:, 0])
        max_h = np.amax(self.step_sizes[:, 1])
        return self._csm[max_v:, max_h:]
    
    @property
    def local_warping_paths(self):
        return self._paths
     
    def calculate_similarity_matrix(self):
        self._sm  = similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma, only_triu=self._symmetric)
        return self._sm
          
    def calculate_cumulative_similarity_matrix(self):
        if self._sm is None:
            self.calculate_similarity_matrix()
        self._csm = cumulative_similarity_matrix(self._sm, tau=self.tau, delta_a=self.delta_a, delta_m=self.delta_m, step_sizes=self.step_sizes, only_triu=self._symmetric)
        return self._csm

    def find_best_paths(self, l_min=10, vwidth=5):
        if self._csm is None:
            self.calculate_cumulative_similarity_matrix()

        # When symmetric, the diagonal is hardcoded (for step_sizes for which the diagonal would not be found first, e.g., [(1, 1), (0, 1), (1, 0)]).
        mask = np.full(self._csm.shape, self._symmetric)
        if self._symmetric:
            # First, mask region around the diagional as if the diagonal is already found as a path.
            mask[np.triu_indices(len(mask), k=vwidth)] = False

        paths = _find_best_paths(self._csm, mask, l_min=l_min, vwidth=vwidth, step_sizes=self.step_sizes)

        if self._symmetric:
            # Prepend the diagonal to the result set.
            diagonal = np.tile(np.arange(len(self.ts), dtype=np.int32), (2, 1)).T
            paths.insert(0, diagonal)

        self._paths = paths
        return self._paths

    @classmethod
    def instance_from_rho(cls, ts, rho, gamma=None, step_sizes=None, ts2=None, equal_weight_dims=False):
        # Make LoCo instance
        loco = cls(ts, gamma=gamma, step_sizes=step_sizes, ts2=ts2, equal_weight_dims=equal_weight_dims)
        # Get the SM, determine tau and delta's
        sm = loco.calculate_similarity_matrix()
        tau = estimate_tau_from_sm(sm, rho, only_triu=loco._symmetric)
        loco.tau = tau
        loco.delta_a = 2 * tau
        loco.delta_m = 0.5
        return loco
    
# Calculate the similarity threshold tau as the rho-quantile of the similarity matrix.
def estimate_tau_from_sm(sm, rho, only_triu=False):
    if only_triu:
        tau = np.quantile(sm[np.triu_indices(len(sm))], rho, axis=None)
    else:
        tau = np.quantile(sm, rho, axis=None)
    return tau

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean))
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False):
    n, m = len(ts1), len(ts2)

    sm = np.full((n, m), -np.inf, dtype=float32)
    for i in range(n):

        j_start = i if only_triu else 0
        j_end   = m
        
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1))
        
        sm[i, j_start:j_end] = similarities

    return sm

@njit(float32[:, :](float32[:, :], float64, float64, float64, int32[:, :], boolean))
def cumulative_similarity_matrix(sm, tau=0.5, delta_a=1.0, delta_m=0.5, step_sizes=np.array([[1, 1], [2, 1], [1, 2]]), only_triu=False):
    n, m = sm.shape
    max_v = np.amax(step_sizes[:, 0])
    max_h = np.amax(step_sizes[:, 1])

    csm = np.zeros((n + max_v, m + max_h), dtype=float32)

    for i in range(n):
        
        j_start = i if only_triu else 0
        j_end   = m
        
        for j in range(j_start, j_end):
            sim     = sm[i, j]

            indices    = np.array([i + max_v, j + max_h]) - step_sizes
            max_cumsim = np.amax(np.array([csm[i_, j_] for (i_, j_) in indices]))

            if sim < tau:
                csm[i + max_v, j + max_h] = max(0, delta_m * max_cumsim - delta_a)
            else:
                csm[i + max_v, j + max_h] = max(0, sim + max_cumsim)
    return csm


@njit(int32[:, :](float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def max_warping_path(csm, mask, i, j, step_sizes=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    path = []
    while i >= max_v and j >= max_h:

        path.append((i - max_v, j - max_h))

        indices = np.array([i, j], dtype=np.int32) - step_sizes

        values = np.array([csm[i_, j_]    for (i_, j_) in indices])
        masked = np.array([mask[i_, j_] for (i_, j_) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        i, j = i - step_sizes[argmax, 0], j - step_sizes[argmax, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32, int32))
def mask_vicinity(path, mask, max_v, max_h, vwidth=10):
    path_ = path + np.array([max_v, max_h])

    # Bresenham's algorithm
    ic, jc = path_[0]
    for (it, jt) in path_[1:]:
        di  =  it - ic
        dj  =  jc - jt
        err = di + dj
        while ic != it or jc != jt:
            mask[ic-vwidth:ic+vwidth+1, jc] = True
            mask[ic, jc-vwidth:jc+vwidth+1] = True
            e = 2 * err
            if e > dj:
                err += dj
                ic  += 1
            if e < di:
                err += di
                jc  += 1

    mask[ic-vwidth:ic+vwidth+1, jc] = True
    mask[ic, jc-vwidth:jc+vwidth+1] = True
    return mask

@njit(numba.types.List(int32[:, :])(float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def _find_best_paths(csm, mask, l_min=10, vwidth=5, step_sizes=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    # Mask all values <= 0
    pos_i, pos_j = np.nonzero(csm <= 0)
    for k in range(len(pos_i)):
        mask[pos_i[k], pos_j[k]] = True

    # Sort indices based on values in D
    pos_i, pos_j = np.nonzero(csm)
    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])
    perm = np.argsort(values)
    sorted_pos_i, sorted_pos_j = pos_i[perm], pos_j[perm]

    k_best = len(sorted_pos_i) - 1
    paths = []
    
    # Keep constructing paths until all positions have been processed
    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)

        # Keep iterating until valid path is constructed
        path_found = False
        while not path_found:

            # Find the best unmasked index in D
            while (mask[sorted_pos_i[k_best], sorted_pos_j[k_best]]):
                k_best -= 1
                if k_best < 0:
                    return paths

            i_best, j_best = sorted_pos_i[k_best], sorted_pos_j[k_best]

            if i_best < max_v or j_best < max_h:
                return paths
            
            # Reconstruct the path
            path = max_warping_path(csm, mask, i_best, j_best, step_sizes=step_sizes)
            # Mask the indices on path
            mask = mask_vicinity(path, mask, max_v, max_h, vwidth=0)

            # Discard if any projection of path is shorter than the minimum motif length  
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        # Path goes through: mask its vicinity
        mask = mask_vicinity(path, mask, max_v, max_h, vwidth)
        
        # Add the path to the result
        paths.append(path)

    return paths
