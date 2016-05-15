from __future__ import division
import numpy as np


def logloss(X_test, y_test, fit):
    pred = fit.predict_proba(X_test)
    f = np.vectorize(
        lambda x: max(min(x, 1-10**-15), 10**-15)
    )
    return -(1/y_test.shape[0])*np.sum(np.sum(y_test*np.log(f(pred))))
