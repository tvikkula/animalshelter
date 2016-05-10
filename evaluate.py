from evaluation import *
import numpy as np
res = np.load('pred-results.npy').T
test = np.load('test-results.npy')
print(res)
print(logloss(test, res))
