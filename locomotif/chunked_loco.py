import numpy as np

import multiprocessing
import gc

from . import loco_jit
from .loco import handle_gamma, ensure_multivariate

import tqdm

# Calculate loco in chunks.
class ChunkedLoCo:

    def __init__(self, 
            T, T2=None,
            tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, gamma=None, equal_weight_dims=False,
            parallel=False, n_processes=1, 
            chunk_overlap=0, chunk_mode="squares", chunk_memory_limit=(1024)**3, # in bytes, default is 1GB
            chunk_begin_rows=None, chunk_end_rows=None, chunk_begin_cols=None, chunk_end_cols=None, # chunk boundaries for custom chunks
        ):

        assert chunk_mode in ["squares", "vertical_stripes"], f"Invalid chunk_mode: {chunk_mode}."

        self._symmetric = False
        if T2 is None:
            self._symmetric = True
            T2 = T

        self.T  = ensure_multivariate(np.array(T, dtype=np.float32))
        self.T2 = ensure_multivariate(np.array(T2, dtype=np.float32))
        assert self.T.shape[1] == self.T2.shape[1], "Input time series must have the same number of dimensions."

        self.gamma = handle_gamma(self.T, gamma, self._symmetric, equal_weight_dims)

        # LoCo args
        self.warping = warping
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m

        # Local warping paths
        self.P = None

        # Parallelization args        
        self._parallel    = parallel
        self._n_processes = n_processes
        if not parallel:
            self._n_processes = 1

        # Divide chunks
        self._chunk_begin_rows, self._chunk_end_rows, self._chunk_begin_cols, self._chunk_end_cols = _divide(len(self.T), len(self.T2), chunk_mode=chunk_mode, chunk_overlap=chunk_overlap, chunk_memory_limit=chunk_memory_limit, chunk_begin_rows=chunk_begin_rows, chunk_end_rows=chunk_end_rows, chunk_begin_cols=chunk_begin_cols, chunk_end_cols=chunk_end_cols)

    @property
    def local_warping_paths(self):
        if self.P is None:
            return None
        
        paths = []
        for (r, c), paths_in_chunk in self.P.items():
            for path in paths_in_chunk:
                # Add the offsets of the chunk
                paths.append(np.vstack((path[:, 0] + self._chunk_begin_rows[r], path[:, 1] + self._chunk_begin_cols[c])).T)
        
        return paths     

    def _to_compute(self, r, c):
        # If symmetric, only the chunks that overlap with the diagonal have to be calculated
        return not self._symmetric or self._chunk_begin_rows[r] < self._chunk_end_cols[c]
        
    def _get_sm_settings(self, r, c):
        if self._symmetric:
            # If symmetric, only the parts above the diagonal have to be calculated
            only_triu   = self._chunk_end_rows[r] > self._chunk_begin_cols[c] + 1
            diag_offset = self._chunk_begin_cols[c] - self._chunk_begin_rows[r]
            return only_triu, diag_offset
        else:
            return False, 0 

    def find_best_paths(self, l_min=None, vwidth=None):
        if l_min is None:
            l_min = min(len(self.T), len(self.T2)) // 10
        if vwidth is None:
            vwidth = l_min // 2

        print(f"Finding paths: Using {self._n_processes} processes.")
        Nc = len(self._chunk_begin_cols)
        Nr = len(self._chunk_begin_rows)
        # Common arguments
        common_args = {'l_min': l_min, 'vwidth': vwidth, 'gamma': self.gamma, 'tau': self.tau, 'delta_a': self.delta_a, 'delta_m': self.delta_m, 'warping': self.warping}
        args = {}

        for r in range(Nr):
            for c in range(Nc):
                if not self._to_compute(r, c):
                    continue
                only_triu, diag_offset = self._get_sm_settings(r, c)
                Tr = self.T[self._chunk_begin_rows[r]:self._chunk_end_rows[r]]
                Tc = self.T2[self._chunk_begin_cols[c]:self._chunk_end_cols[c]]
                args[r, c] = {'Tr': Tr, 'Tc': Tc} | common_args | {'only_triu': only_triu, 'diag_offset': diag_offset}
        
        if self._parallel:
            with multiprocessing.Pool(processes=self._n_processes) as pool:
                paths = list(tqdm.tqdm(
                            pool.imap(_find_paths_wrapper, [list(arg.values()) for arg in args.values()]),
                            total=len(args),
                            desc="Processing chunks"
                        ))
        else:
            paths = [_find_paths_wrapper(arg.values()) for arg in tqdm.tqdm(args.values(), total=len(args), desc="Processing chunks")]

        self.P = {}
        for chunk_idx, (r, c) in enumerate(args.keys()):
            self.P[(r, c)] = paths[chunk_idx]
            
        return self.P

    @classmethod
    def instance_from_rho(cls, 
            T, rho, T2=None,
            delta_m=0.5, warping=True, gamma=None, equal_weight_dims=False, 
            parallel=False, n_processes=1,
            chunk_overlap=0, chunk_mode="squares", chunk_memory_limit=(1024)**3,
            precision_tau_estimation=0.001
        ):

        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5
        
        loco = cls(
            T, T2=T2,
            tau=None, delta_a=None, delta_m=delta_m, warping=warping, gamma=gamma, equal_weight_dims=equal_weight_dims, 
            parallel=parallel, n_processes=n_processes,
            chunk_mode=chunk_mode, chunk_overlap=chunk_overlap, chunk_memory_limit=chunk_memory_limit
        )
        print(f"Estimating tau. Using {n_processes} processes.")
        # We estimate the distribution of the simililarities in the full similarity matrix using a histogram.
        # This allows us to calculate the rho-quantile of the full similarity matrix with a certain precision (equal to the bin width)
        bin_width = precision_tau_estimation
        n_bins = int(np.ceil(1 / bin_width))
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Here, no overlap
        chunk_begin_rows, chunk_end_rows, chunk_begin_cols, chunk_end_cols = _divide(len(loco.T), len(loco.T2), chunk_mode=chunk_mode, chunk_overlap=0, chunk_memory_limit=chunk_memory_limit)

        Nc = len(chunk_begin_cols)
        Nr = len(chunk_begin_rows)
        args = {}
        n_chunks = 0
        for r in range(Nr):
            for c in range(Nc):
                if not loco._to_compute(r, c):
                    continue
                only_triu, diag_offset = loco._get_sm_settings(r, c)
                Tr = loco.T[chunk_begin_rows[r]:chunk_end_rows[r]]
                Tc = loco.T2[chunk_begin_cols[c]:chunk_end_cols[c]]
                args[r, c] = {'Tr': Tr, 'Tc': Tc, 'gamma': loco.gamma, 'only_triu': only_triu, 'diag_offset': diag_offset, 'n_bins': n_bins}
                n_chunks += 1

        if parallel:
            with multiprocessing.Pool(processes=n_processes) as pool:
                histograms = list(tqdm.tqdm(
                            pool.imap(_compute_histogram_from_sm, [list(arg.values()) for arg in args.values()]),
                            total=len(args),
                            desc="Processing chunks"
                        ))
        else:
            histograms = [_compute_histogram_from_sm(arg.values()) for arg in tqdm.tqdm(args.values(), total=len(args), desc="Processing chunks")]
        # Estimate tau from the histogram
        histogram = np.sum(histograms, axis=0)
        histogram = histogram / np.sum(histogram)
        tau = bin_centers[np.searchsorted(np.cumsum(histogram), rho)]
        print(f"Estimated tau: {tau}")
        loco.tau = tau
        loco.delta_a = 2*tau
        return loco
    
def _divide(n, m, chunk_mode="squares", chunk_overlap=0, chunk_memory_limit=(1024)**3, verbose=False, chunk_begin_rows=None, chunk_end_rows=None, chunk_begin_cols=None, chunk_end_cols=None):

    if chunk_begin_rows is not None and chunk_end_rows is not None and chunk_begin_cols is not None and chunk_end_cols is not None:
        assert len(chunk_begin_rows) == len(chunk_end_rows) and len(chunk_begin_cols) == len(chunk_end_cols)
        chunk_widths  = (chunk_end_cols - chunk_begin_cols)
        chunk_heights = (chunk_end_rows - chunk_begin_rows)
        fit = 4 * (chunk_widths * chunk_heights) <= chunk_memory_limit
        assert np.all(fit), f"chunk of size {(chunk_heights[np.where(~fit)[0][0]], chunk_widths[np.where(~fit)[0][0]])} does not fit within memory limit of {chunk_memory_limit} bytes"
        return chunk_begin_rows, chunk_end_rows, chunk_begin_cols, chunk_end_cols

    if chunk_mode == "squares":
        chunk_width = chunk_height = int(np.floor(np.sqrt(chunk_memory_limit / 4)))
        assert chunk_height > chunk_overlap, "chunk_size must be larger than chunk_overlap"
        chunk_begin_cols = np.arange(0, m - chunk_overlap, chunk_height - chunk_overlap, dtype=int)
        chunk_end_cols = np.minimum(chunk_begin_cols + chunk_height, m)
        chunk_begin_rows = np.arange(0, n - chunk_overlap, chunk_height - chunk_overlap, dtype=int)
        chunk_end_rows = np.minimum(chunk_begin_rows + chunk_height, n)

    elif chunk_mode == "vertical_stripes":
        chunk_width = int(np.ceil(chunk_memory_limit / (4*n)))
        assert chunk_width > chunk_overlap, "chunk_width must be larger than chunk_overlap"
        chunk_height = n
        chunk_begin_cols = np.arange(0, m-chunk_overlap, chunk_width-chunk_overlap, dtype=int)
        chunk_end_cols = np.minimum(chunk_begin_cols + chunk_width, m)
        chunk_begin_rows = np.array([0], dtype=int)
        chunk_end_rows = np.array([n], dtype=int)

    if verbose:
        print(f"Divided into {len(chunk_begin_cols)*len(chunk_begin_rows)} chunks of size ({chunk_height}, {chunk_width})")

    return chunk_begin_rows, chunk_end_rows, chunk_begin_cols, chunk_end_cols

def _find_paths_wrapper(args):
    Tr, Tc, l_min, vwidth, gamma, tau, delta_a, delta_m, warping, only_triu, diag_offset = args
    sm = loco_jit.similarity_matrix_ndim(Tr, Tc, gamma=gamma, only_triu=only_triu, diag_offset=diag_offset)
    if warping:
        csm = loco_jit.cumulative_similarity_matrix_warping(sm=sm, tau=tau, delta_a=delta_a, delta_m=delta_m, only_triu=only_triu, diag_offset=diag_offset)
    else:
        csm = loco_jit.cumulative_similarity_matrix_no_warping(sm=sm, tau=tau, delta_a=delta_a, delta_m=delta_m, only_triu=only_triu, diag_offset=diag_offset)
    del sm
    gc.collect()
    mask = np.full(csm.shape, False)
    ## TODO: CSM can be removed after sorting the start indices (only the order of non-zero cells matters).
    P = loco_jit.find_best_paths(csm, mask, tau=tau, l_min=l_min, vwidth=vwidth, warping=warping)
    P = [path-2 for path in P]
    del csm
    del mask
    gc.collect()
    return P

def _compute_histogram_from_sm(args):
    Tr, Tc, gamma, only_triu, diag_offset, n_bins = args
    sm = loco_jit.similarity_matrix_ndim(Tr, Tc, gamma=gamma, only_triu=only_triu, diag_offset=diag_offset)

    chunk_height, chunk_width = sm.shape

    if only_triu:
        similarities = sm[np.triu_indices(n=chunk_height, m=chunk_width, k=-diag_offset)].flatten() 
    else:
        similarities = sm.flatten() 

    indices = (similarities * n_bins).astype(np.int32)
    indices[indices == n_bins] -= 1
    histogram = np.bincount(indices, minlength=n_bins)

    assert np.sum(histogram) == np.sum(~np.isinf(sm))
    return histogram