from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pprint
pp = pprint.PrettyPrinter(indent=4)


def train(X_train, y_train):
    abc = AdaBoostClassifier(
        RandomForestClassifier(
            bootstrap=True, compute_importances=None,
            criterion='gini', max_depth=None, max_features='log2',
            max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
            min_samples_split=50, n_estimators=15, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0
        ),
        n_estimators=100,
        learning_rate=0.2
    )
    fit = abc.fit(X_train, y_train)
    return fit


def test(pred, y_test):
    return accuracy_score(pred, y_test)