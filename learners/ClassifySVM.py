#!/usr/bin/python                                                               
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import numpy as np


def train(X_train, y_train, svc = None):
    if (svc == None):
        svc = SVC(kernel = 'linear', gamma = 0.0005, C = 500)
    fit = svc.fit(X_train, y_train)
    return fit


def test(X_test, y_test, fit):
    pred = fit.predict(X_test)
    return accuracy_score(pred, y_test)


def gridsearch(X_train, y_train):
    svc = SVC(class_weight='auto', probability=True)
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 50, 1e2, 5e2, 1e3, 5e3],
        'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 10]
    }
    clf = GridSearchCV(svc, param_grid, scoring='log_loss')
    clf.fit(X_train, y_train)
    scores = clf.grid_scores_
    # Sort by mean (note, it's using namedtuples)
    scores.sort(key=lambda x:x.mean_validation_score, reverse=True)
    print(clf.best_estimator_)
    return clf.best_estimator_, scores


def plotGridScores(scores):
    # Strip 0 validation scores away:
    newScores = []
    for score in scores:
        if (score.mean_validation_score > 0.0):
            newScores.append(score)
        else:
            # Scores are sorted, we can break if we find a 0-value
            break
    if (len(newScores) > 30):
        newScores = newScores[0:30]
    from matplotlib import pyplot as plt
    plt.bar(range(0,len(newScores),1), [score.mean_validation_score for score in newScores])
    plt.xticks(np.arange(0.5, len(newScores), 1),
               [str(score.parameters.values()) for score in newScores], rotation=90)
    plt.xlabel('SVM params [kernel, C, gamma]')
    plt.ylabel('F1 score')
    plt.title('Grid search results for SVM')
    plt.legend()
    plt.ylim(newScores[-1].mean_validation_score, newScores[0].mean_validation_score * 1.01)
    plt.show()