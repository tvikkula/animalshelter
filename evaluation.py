from __future__ import division
import numpy as np

def logloss(Y, pred):
    f = np.vectorize(
        lambda x: max(min(x, 1-10**-15), 10**-15)
    )
    return -(1/Y.shape[0])*np.sum(Y*np.log(f(pred.T)))
