#!/usr/bin/python                                                               
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train):
    svc = SVC(kernel = 'poly', gamma = 100.0, C = 100.0)
    fit = svc.fit(features_train, labels_train)
    return fit

def accuracy(features_test, labels_test, fit):
    pred = fit.predict(features_test)
    return accuracy_score(pred, labels_test)


