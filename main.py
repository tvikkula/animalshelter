import sys
sys.path.append('learners/')
import pandas as pd
import RFClassifier
import numpy as np

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

#    .join(pd.get_dummies(test.Breed_formatted, prefix='is'))\

labels_test = pd.DataFrame(columns=labels_train.columns, index = np.arange(len(features_test.index)))
for label in labels:
    labels_test[label] = 0

features_test.to_csv('data/test_data.csv', sep=',', encoding='utf-8')
labels_train.to_csv('data/train_labels.csv', sep=',', enconding='utf-8')
features_train.to_csv('data/train_data.csv', sep=',', encoding='utf-8')

print('Creating a fit from training data')

fit = RFClassifier.train(features_train, labels_train)

print('Evaluating accuracy')
acc = RFClassifier.test(features_test, labels_test, fit)
print acc
