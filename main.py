import sys
sys.path.append('learners/')
import pandas as pd
from ClassifySVM import *


def logloss(output):
    '''
    Return the logloss evaluator for the given output values

    Params
    output:    Output probabilities as a DF.

    Returns
    logloss
    '''
    return -(1/nrow(output))*output.sum(axis=1)

def output(test):
    '''
    Return a DF with only the predicted columns

    Params
    test:      Output test Dataframe

    Returns:
    output:    Output dataframe with only the predicted cols.
    '''
    return test[[
            'Return_to_owner',
            'Euthanasia',
            'Adoption',
            'Transfer',
            'Died'
            ]]

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

features_for_train = list(features)
features_for_train.append(predicted)

print('Creating feature and label sets to test and train')
features_train = train[features_for_train]
print(features_train.columns.values)

# Create dummies:
features_train = features_train\
    .append(pd.get_dummies(features_train.isMix))\
    .append(pd.get_dummies(features_train.Sex))\
    .append(pd.get_dummies(features_train.Neutered))\
    .append(pd.get_dummies(features_train.Breed_formatted))\
    .append(pd.get_dummies(features_train.OutcomeType))

labels_train = features_train[labels]
print(features)
features_test = test[features]

features_test = features_test\
    .append(pd.get_dummies(features_test.isMix))\
    .append(pd.get_dummies(features_test.Sex))\
    .append(pd.get_dummies(features_test.Neutered))\
    .append(pd.get_dummies(features_test.Breed_formatted))\

for label in labels:
    features_test[label] = 0

labels_test = features_test[labels]
features_test.to_csv('data/test_data.csv', sep=',', encoding='utf-8')
features_train.to_csv('data/train_data.csv', sep=',', encoding='utf-8')

print('Creating a fit from training data')
fit = classify(features_train, labels_train)

print('Evaluating accuracy')
acc = accuracy(features_test, labels_test, fit)
print acc
