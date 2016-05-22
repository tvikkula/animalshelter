#!/usr/bin/python
from xgboost import xgb


def train(X_train, y_train):
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300,
                            learning_rate=0.1)
    return gbm.fit(X_train, y_train)
