from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pprint
pp = pprint.PrettyPrinter(indent=4)


def train(features_train, labels_train):
    abc = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=20,
        learning_rate=1
    )
    print features_train.shape
    print labels_train.shape
    fit = abc.fit(features_train, labels_train)
    return fit


def test(pred, labels_test):
    return accuracy_score(pred, labels_test)