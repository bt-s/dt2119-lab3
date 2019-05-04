import numpy as np


def get_data(data, feature="lmfcc", dynamic=False):
    pass

def dynamic_features(lmfcc):
    N = lmfcc.shape[0]
    M = lmfcc.shape[1] * 7
    dyn = np.zeros((N, M))

    lmfcc = np.vstack((np.flip(lmfcc[1:4, :], 0),
                      lmfcc,
                      np.flip(lmfcc[-4:-1, :], 0)))
    for i in range(N):
        dyn[i, :] = lmfcc[i:i+7].flatten()

    return dyn
