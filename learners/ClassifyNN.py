#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import numpy as np


def train(X_train, y_train, svc = None):
    if (clf == None):
        clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    fit = clf.fit(X_train, y_train)
    return fit


def test(X_test, y_test, fit):
    pred = fit.predict(X_test)
    return accuracy_score(pred, y_test)
