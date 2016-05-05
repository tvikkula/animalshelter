import numpy as np

def evaluation(Y, pred):
    -(1/Y.shape[0])*np.sum(Y*math.log(pred))
