import sys
from learners import PCA, RFClassifier, ABClassifier, ClassifySVM, XGBClassifier
from submission import *
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)


def read_data(test):
    Y = pd.read_csv('data/train_labels.csv').values.ravel()
    X = pd.read_csv('data/train_data.csv')
    if test:
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state = 42
        )
        # dummify y_test:
        y_test = pd.DataFrame(y_test)
        y_test = pd.get_dummies(y_test)
        return X_train, y_train, X_test, y_test
    else:
        X_test = pd.read_csv('data/test_data.csv')
        y_test = None
        return X, Y, X_test, y_test


def xgb(test):
    X, Y, X_test, y_test = read_data(test)
    fit = XGBClassifier.train(X, Y)
    if test:
        from evaluation import logloss
        print(logloss(X_test, y_test, fit))
    else:
        submission(X_test, fit)


def ada(test):
    X, Y, X_test, y_test = read_data(test)
    fit = ABClassifier.train(X, Y)
    if test:
        from evaluation import logloss
        print(logloss(X_test, y_test, fit))
    else:
        submission(X_test, fit)


def rf(test):
    X, Y, X_test, y_test = read_data(test)

    # Can you training set for hyperparam validation
    # in Sklearn for certain algs, including RandomForestClassification.

    pca = PCA.doPCA(X, n=4)
    X_pca = pca.transform(X)
    X_test_pca = pca.transform(X_test)
    '''
    best_fit, rf_grid_scores = RFClassifier.gridsearch(
        X_pca, Y, n = 100
    )
    '''

    fit = RFClassifier.train(X_pca, Y)
    if test:
        from evaluation import logloss
        print(logloss(X_test_pca, y_test, fit))
    else:
        submission(X_test_pca, fit)


def svm(test):
    X, Y, X_test, y_test = read_data(test)
    '''
    best_fit, rf_grid_scores = ClassifySVM.gridsearch(
        X, Y
    )
    '''
    fit = ClassifySVM.train(X, Y)
    if test:
        from evaluation import logloss
        print(logloss(X_test, y_test, fit))
    else:
        submission(X_test, fit)


def main(argv):
    if len(argv) > 2 and argv[2] == 'test':
        test = True
    else:
        test = False
    if len(argv) > 1 and argv[1] == 'rf':
        rf(test)
    elif len(argv) > 1 and argv[1] == 'ada':
        ada(test)
    else:
        xgb(test = True)


if __name__ == "__main__":
    main(sys.argv)
