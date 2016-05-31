from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import pprint
pp = pprint.PrettyPrinter(indent=4)


def train(X_train, y_train, rfc = None):
    if (rfc == None):
        # RFC that is found previously using GridSearch:
        rfc = RandomForestClassifier(
            bootstrap=True,
            criterion='gini', max_depth=None, max_features='log2',
            max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=50, n_estimators=400, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0
        )
    fit = rfc.fit(X_train, y_train)
    return fit 


def test(pred, y_test):
    return accuracy_score(pred, y_test)


def gridsearch(X_train, y_train, n):
    clf = RandomForestClassifier(
        n_estimators = n,
        n_jobs = -1
    )
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'min_samples_split': [1, 2, 5, 10, 20, 50],
        'max_features': ['sqrt', 'log2', None]
    }
    clf = GridSearchCV(clf, param_grid, scoring='log_loss')
    clf.fit(X_train, y_train)
    scores = clf.grid_scores_
    # Sort by mean (note, it's using namedtuples)
    scores.sort(key=lambda x:x.mean_validation_score, reverse=True)
    print(clf.best_estimator_)
    return clf.best_estimator_, scores
