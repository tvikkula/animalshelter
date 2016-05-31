#!/usr/bin/python
import xgboost as xgb


def train(X_train, y_train):
    gbm = xgb.XGBClassifier(max_depth=6, n_estimators=100,
                            learning_rate=0.1, gamma=0.0,
                            subsample=1.0, colsample_bytree=0.5,
                            eval_metric=logloss, objective=multi:softmax)
    return gbm.fit(X_train, y_train)
