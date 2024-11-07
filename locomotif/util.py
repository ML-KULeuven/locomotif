import numpy as np


def is_znormalized(ts, tolerance=0.01):
    assert ts.ndim == 1 or ts.ndim == 2
    mean = np.mean(ts, axis=None)
    std = np.std(ts, axis=None)

    znormalized = (np.abs(mean) < tolerance) and (np.abs(std - 1) < tolerance)
    if ts.ndim == 2 and not znormalized:
        # A multidimensional time series is also znormalized if each of its dimensions is znormalized
        _, ndim = ts.shape
        for d in range(ndim):
            if not is_znormalized(ts[:, d]):
                return False
        return True
        
    return znormalized


def is_unitstd(ts, tolerance=0.01):
    assert ts.ndim == 1 or ts.ndim == 2
    # mean = np.mean(ts, axis=None)
    std = np.std(ts, axis=None)

    unitstd = (np.abs(std - 1) < tolerance)
    if ts.ndim == 2 and not unitstd:
        # Even if a multidimensional time series doesn't have a std of 1, it's OK if each of its dimensions has a std of 1
        _, ndim = ts.shape
        for d in range(ndim):
            if not is_unitstd(ts[:, d]):
                return False
        return True
        
    return unitstd