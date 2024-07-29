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