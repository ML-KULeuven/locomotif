import numpy as np

from . import loco_jit

class LoCo:

    def __init__(self, ts, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, ts2=None, equal_weight_dims=False):
        
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
        self.warping = warping
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
        return self._csm[2:, 2:]
    
    @property
    def local_warping_paths(self):
        return self._paths
     
    def calculate_similarity_matrix(self):
        self._sm  = similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma, only_triu=self._symmetric, diag_offset=0)
        return self._sm
          
    def calculate_cumulative_similarity_matrix(self):
        if self._sm is None:
            self.calculate_similarity_matrix()            
        self._csm = cumulative_similarity_matrix(self._sm, tau=self.tau, delta_a=self.delta_a, delta_m=self.delta_m, warping=self.warping, only_triu=self._symmetric, diag_offset=0)
        return self._csm

    def find_best_paths(self, l_min=10, vwidth=5):
        if self._csm is None:
            self.calculate_cumulative_similarity_matrix()

        # When symmetric, the diagonal is hardcoded (TODO: can be removed as step_sizes is not configurable anymore)
        mask = np.full(self._csm.shape, self._symmetric)
        if self._symmetric:
            # First, mask region around the diagional as if the diagonal is already found as a path.
            mask[np.triu_indices(len(mask), k=vwidth+1)] = False

        paths = find_best_paths(self._csm, mask, self.tau, l_min=l_min, vwidth=vwidth, warping=self.warping)
        paths = [path-2 for path in paths]

        if self._symmetric:
            # Prepend the diagonal to the result set.
            diagonal = np.tile(np.arange(len(self.ts), dtype=np.int32), (2, 1)).T
            paths.insert(0, diagonal)

        self._paths = paths
        return self._paths

    @classmethod
    def instance_from_rho(cls, ts, rho, gamma=None, warping=True, ts2=None, equal_weight_dims=False):
        # Make LoCo instance
        loco = cls(ts, gamma=gamma, warping=warping, ts2=ts2, equal_weight_dims=equal_weight_dims)
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

def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    return loco_jit.similarity_matrix_ndim(ts1, ts2, gamma, only_triu, diag_offset)

def cumulative_similarity_matrix(sm, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, only_triu=False, diag_offset=0):
    if warping:
        return loco_jit.cumulative_similarity_matrix_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    else:
        return loco_jit.cumulative_similarity_matrix_no_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)

def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True):
    paths = loco_jit.find_best_paths(csm, mask, tau, l_min, vwidth, warping)
    return paths