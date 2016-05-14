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


def submission(test, fit):
    y_pred = fit.predict_proba(test)

    results = pd.DataFrame(
        {
            'ID': range(1, len(y_pred) + 1),
            'Adoption': y_pred[:,0],
            'Died': y_pred[:,1],
            'Euthanasia': y_pred[:,2],
            'Return_to_owner': y_pred[:,3],
            'Transfer': y_pred[:,4]
        }
    )

    results.to_csv("submission.csv", index=False)


def rfclassify(X_train, y_train, X_validate, y_validate, X_test):
    print('Hyperparam validation')
    best_fit, rf_grid_scores = RFClassifier.gridsearch(
        X_validate, y_validate, n = 500
    )

    print(best_fit)

    print('Creating a fit from training data')
    fit = RFClassifier.train(X_train, y_train, best_fit)

    print('Evaluating accuracy')
    pred = fit.predict_proba(X_test)
    res = fit.predict(X_test)
    print type(res)
    return pred, res, best_fit
