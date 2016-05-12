import sys
sys.path.append('learners/')
import RFClassifier
import ABClassifier
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)


def adaclassify(X_train, y_train, X_test):
    fit = ABClassifier.train(X_train, y_train)
    pred = fit.predict_proba(X_test)
    res = fit.predict(X_test)
    return pred, res, fit


def submission(best_fit, X, y, test):
    fit = ABClassifier.train(X, y, best_fit)
    y_pred = fit.predict_proba(test)
    print fit.n_outputs
    print fit.n_classes_
    print fit.classes_
    print len(y_pred)
    pp.pprint(y_pred)

    results = pd.read_csv("data/sample_submission.csv")

    results['Adoption'],\
        results['Died'],\
        results['Euthanasia'],\
        results['Return_to_owner'],\
        results['Transfer'] = \
            y_pred[:,0],\
            y_pred[:,1],\
            y_pred[:,2],\
            y_pred[:,3],\
            y_pred[:,4]

    results.to_csv("submission.csv", index=False)


def rfclassify(X_train, y_train, X_validate, y_validate, X_test):
    print('Hyperparam validation')
    best_fit, rf_grid_scores = RFClassifier.gridsearch(
        X_validate, y_validate, n = 100
    )

    print(best_fit)

    print('Creating a fit from training data')
    fit = RFClassifier.train(X_train, y_train, best_fit)

    print('Evaluating accuracy')
    pred = fit.predict_proba(X_test)
    res = fit.predict(X_test)
    print type(res)
    return pred, res, best_fit
