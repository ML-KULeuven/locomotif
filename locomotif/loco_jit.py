import numpy as np


### JIT
from numba import int32, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32))
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)

    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end   = m
        
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1))
        
        sm[i, j_start:j_end] = similarities

    return sm

@njit
def max3(a, b, c):
    if a >= b:
        if a >= c:
            return a
        else:
            return c
    else:
        if b >= c:
            return b
        else:
            return c
        
@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            max_cs = max3(csm[i - 1 + 2, j - 1 + 2], csm[i - 2 + 2, j - 1 + 2], csm[i - 1 + 2, j - 2 + 2])

            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * max_cs - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + max_cs)
    return csm

@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * csm[i - 1 + 2, j - 1 + 2] - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + csm[i - 1 + 2, j - 1 + 2])

    return csm


@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32))
def best_path_warping(csm, mask, i, j):
    
    path = []
    while i >= 2 and j >= 2:

        path.append((i, j))

        maximum = max3(csm[i - 1, j - 1], csm[i - 2, j - 1], csm[i - 1, j - 2])

        if csm[i - 1, j - 1] == maximum:
            if mask[i - 1, j - 1]:
                break
            i, j = i - 1, j - 1
        elif csm[i - 2, j - 1] == maximum:
            if mask[i - 2, j - 1]:
                break
            i, j = i - 2, j - 1
        else:
            if mask[i - 1, j - 2]:
                break
            i, j = i - 1, j - 2

    path.reverse()
    return np.array(path, dtype=np.int32)

@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32))
def best_path_no_warping(csm, mask, i, j):
    
    path = []
    while i >= 2 and j >= 2:

        path.append((i, j))

        if mask[i - 1, j - 1]:
            break

        i, j = i - 1, j - 1

    path.reverse()
    return np.array(path, dtype=np.int32)


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32))
def mask_vicinity(path, mask, vwidth=10):

    n, m = mask.shape
    
    for k in range(len(path)-1):
        ic, jc = path[k]
        it, jt = path[k + 1]
        
        di, dj = (it - ic, jt - jc)
        
        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
        j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
        
        mask[i1 : i2, jc] = True
        mask[ic, j1 : j2] = True
                
        if di == 2 and dj == 1:
            if i2 + 1 < n:
                mask[ic + 1, jc] = True
            mask[ic + 1, j1 : j2] = True
            
        elif di == 1 and dj == 2:
            if j2 + 1 < m:
                mask[ic, jc + 1] = True
            mask[i1 : i2, jc + 1] = True
            
        else:
            if not (di == 1 and dj == 1):
                raise Exception("Path does not comply to the allowed step sizes")

    (ic, jc) = path[-1]
    mask[max(0, ic - vwidth) : min(n, ic + vwidth + 1), jc] = True
    mask[ic, max(0, jc - vwidth) : min(m, jc + vwidth + 1)] = True
    return mask


@njit
def low16_sort(score_bits, pos_i, pos_j, temp_bits, temp_pos_i, temp_pos_j, size):
    for shift in (0, 8):
        counts = np.zeros(256, dtype=np.int64)
        for k in range(size):
            byte = (score_bits[k] >> shift) & np.uint32(255)
            counts[byte] += 1

        total = 0
        for byte in range(256):
            count = counts[byte]
            counts[byte] = total
            total += count

        next_index = counts
        for k in range(size):
            byte = (score_bits[k] >> shift) & np.uint32(255)
            index = next_index[byte]
            temp_bits[index] = score_bits[k]
            temp_pos_i[index] = pos_i[k]
            temp_pos_j[index] = pos_j[k]
            next_index[byte] = index + 1

        score_bits, temp_bits = temp_bits, score_bits
        pos_i, temp_pos_i = temp_pos_i, pos_i
        pos_j, temp_pos_j = temp_pos_j, pos_j

    return pos_i, pos_j

@njit
def split_into_buckets(score_bits, pos_i, pos_j):
    bucket_counts = np.zeros(65536, dtype=np.int64)
    for k in range(len(score_bits)):
        bucket = score_bits[k] >> np.uint32(16)
        bucket_counts[bucket] += 1

    bucket_starts = np.empty(65537, dtype=np.int64)
    total = 0
    max_bucket_size = 0
    for bucket in range(65536):
        bucket_starts[bucket] = total
        total += bucket_counts[bucket]
        max_bucket_size = max(max_bucket_size, bucket_counts[bucket])
    bucket_starts[65536] = total

    next_index = bucket_starts[:65536].copy()
    bits_by_bucket = np.empty(len(score_bits), dtype=np.uint32)
    pos_i_by_bucket = np.empty(len(score_bits), dtype=np.int32)
    pos_j_by_bucket = np.empty(len(score_bits), dtype=np.int32)
    for k in range(len(score_bits)):
        bucket = score_bits[k] >> np.uint32(16)
        index = next_index[bucket]
        bits_by_bucket[index] = score_bits[k]
        pos_i_by_bucket[index] = pos_i[k]
        pos_j_by_bucket[index] = pos_j[k]
        next_index[bucket] = index + 1

    return bucket_starts, bits_by_bucket, pos_i_by_bucket, pos_j_by_bucket, max_bucket_size

@njit(List(Array(int32, 2, 'C'))(float32[:, :], boolean[:, :], float32, int32, int32, boolean))
def find_best_paths(csm, mask, minimum_score, l_min=10, vwidth=5, warping=True):
    # Mask all zeros
    mask = mask | (csm <= 0)
    
    start_mask = (~mask) & (csm >= minimum_score)
    
    pos_i, pos_j = np.nonzero(start_mask)
    
    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])
    score_bits = values.view(np.uint32)
    bucket_starts, bits_by_bucket, pos_i_by_bucket, pos_j_by_bucket, max_bucket_size = split_into_buckets(score_bits, pos_i, pos_j)

    bucket_bits = np.empty(max_bucket_size, dtype=np.uint32)
    bucket_pos_i = np.empty(max_bucket_size, dtype=np.int32)
    bucket_pos_j = np.empty(max_bucket_size, dtype=np.int32)
    temp_bits = np.empty(max_bucket_size, dtype=np.uint32)
    temp_pos_i = np.empty(max_bucket_size, dtype=np.int32)
    temp_pos_j = np.empty(max_bucket_size, dtype=np.int32)

    paths = []

    for bucket in range(65535, -1, -1):
        size = 0
        for k in range(bucket_starts[bucket], bucket_starts[bucket + 1]):
            i, j = pos_i_by_bucket[k], pos_j_by_bucket[k]
            if mask[i, j]:
                continue
            bucket_bits[size] = bits_by_bucket[k]
            bucket_pos_i[size] = i
            bucket_pos_j[size] = j
            size += 1

        if size == 0:
            continue

        sorted_pos_i, sorted_pos_j = low16_sort(bucket_bits, bucket_pos_i, bucket_pos_j, temp_bits, temp_pos_i, temp_pos_j, size)
        for k_best in range(size - 1, -1, -1):
            i_best, j_best = sorted_pos_i[k_best], sorted_pos_j[k_best]
            if mask[i_best, j_best]:
                continue

            if i_best < 2 or j_best < 2:
                return paths
            
            if warping:
                path = best_path_warping(csm, mask, i_best, j_best)
            else:
                path = best_path_no_warping(csm, mask, i_best, j_best)
                
            mask = mask_vicinity(path, mask, 0)
            # mask = mask_path(path, mask)
            
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                mask = mask_vicinity(path, mask, vwidth)
                paths.append(path)

    return paths
