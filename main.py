import sys
sys.path.append('learners/')
import pandas as pd
import RFClassifier
import numpy as np
from sklearn.cross_validation import train_test_split

print('Reading data to test and train sets')
train = pd.read_csv('data/train_cleaned.csv')
test = pd.read_csv('data/test_cleaned.csv')

labels = [
    'Return_to_owner',
    'Euthanasia',
    'Adoption',
    'Transfer',
    'Died'
    ]

features = [
    'isMix',
    'Breed_formatted',
    'Sex',
    'Neutered'
    ]

predicted = 'OutcomeType'

print('Creating feature and label sets to test and train')

# Create dummies:
features_train = pd.get_dummies(train.isMix, prefix='is')\
    .join(pd.get_dummies(train.Sex, prefix='is'))\
    .join(pd.get_dummies(train.Neutered, prefix='is'))

#    .join(pd.get_dummies(train.Breed_formatted, prefix='is'))

labels_train = pd.get_dummies(train.OutcomeType)

features_test = pd.get_dummies(test.isMix, prefix='is')\
    .join(pd.get_dummies(test.Sex, prefix='is'))\
    .join(pd.get_dummies(test.Neutered, prefix='is'))

#    .join(pd.get_dummies(test.Breed_formatted, prefix='is'))

labels_test = pd.DataFrame(columns=labels_train.columns, index = np.arange(len(features_test.index)))
for label in labels:
    labels_test[label] = 0

features_test.to_csv('data/test_data.csv', sep=',', encoding='utf-8')
labels_train.to_csv('data/train_labels.csv', sep=',', enconding='utf-8')
features_train.to_csv('data/train_data.csv', sep=',', encoding='utf-8')

print('Creating training and validation sets')
X = features_train
Y = labels_train
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state = 42
)

X_train, X_validate, y_train, y_validate = train_test_split(
    X_train, y_train, test_size = 0.3, random_state = 42
)

print('Hyperparam validation')
best_fit, rf_grid_scores = RFClassifier.gridsearch(
    X_validate, y_validate, n = 5
)
print(rf_grid_scores)
print(best_fit)

print('Creating a fit from training data')
fit = RFClassifier.train(X_train, y_train, best_fit)
print(type(y_test))
print('Evaluating accuracy')
pred = fit.predict(X_test)
acc = RFClassifier.test(pred, y_test)
print acc
np.save('pred-results', pred)
np.save('test-results', y_test)
