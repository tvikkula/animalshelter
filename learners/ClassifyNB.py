#!/usr/bin/python
from sklearn.naive_bayes import GaussianNB
import numpy as np


def train(X_train, y_train):
    gnb = GaussianNB()
    fit = gnb.fit(X_train, y_train)
    return fit


def test(X_test, y_test, fit):
    pred = fit.predict(X_test)
    return np.sum(pred == y_test) / float(len(pred))
