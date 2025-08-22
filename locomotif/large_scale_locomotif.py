from .chunked_locomotif import ChunkedLoCoMotif
from .chunked_loco import ChunkedLoCo

class LargeScaleLoCoMotif(ChunkedLoCoMotif):

    def __init__(
            self, 
            T, 
            l_min, l_max, 
            gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, equal_weight_dims=False,
            chunk_memory_limit=(1024)**3, # in bytes, default is 1GB 
            parallel=False, n_processes=1,
        ):
        super().__init__(T, l_min, l_max, parallel=parallel, n_processes=n_processes)
        # We pass T as T2 here such that the chunks are fully computed. 
        # An alternative is to only process the upper triangular of the full SSM and then mirror the obtained paths (half the work).
        self._loco = ChunkedLoCo( 
            T, T2=T,
            gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, warping=warping, equal_weight_dims=equal_weight_dims,
            chunk_overlap=l_max, chunk_mode="vertical_stripes", chunk_memory_limit=chunk_memory_limit, # in bytes, default is 10MB
            parallel=self._parallel, n_processes=self._n_processes, 
        )
        
    @classmethod
    def instance_from_rho(cls, T, l_min, l_max, rho=None, warping=True, equal_weight_dims=False, chunk_memory_limit=10*(1024)**2, parallel=False, n_processes=1):
        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5
        lcm = cls(T, l_min, l_max, equal_weight_dims=equal_weight_dims, chunk_memory_limit=chunk_memory_limit, parallel=parallel, n_processes=n_processes)
        lcm._loco = ChunkedLoCo.instance_from_rho(T, rho, T2=T, warping=warping, parallel=parallel, n_processes=n_processes, chunk_overlap=l_max, chunk_mode="vertical_stripes", chunk_memory_limit=chunk_memory_limit) 
        return lcm
    
    def find_best_paths(self):        
        super().find_best_paths()
    
    def find_best_motif_sets(self, nb=None, overlap=0.25):
        return super().find_best_motif_sets(nb, overlap)