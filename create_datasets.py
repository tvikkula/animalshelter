import pandas as pd

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

categories = [
    'isMix',
    'Breed_formatted',
    'Sex',
    'Neutered',
    'AnimalType'
    ]

predicted = 'OutcomeType'

print('Creating feature and label sets to test and train')

labels_train = train['OutcomeType']

train = train[categories]
test = test[categories]

train["Train"] = 1
test["Train"] = 0

combined = pd.concat([train, test])
combined_dummies = pd.get_dummies(combined, columns=categories)

train = combined_dummies[combined_dummies["Train"] == 1]
test = combined_dummies[combined_dummies["Train"] == 0]
train = train.drop(["Train"], axis=1)
test = test.drop(["Train"], axis=1)

print train.head(1)
print test.head(1)
print labels_train.head(1)
print labels_train.shape
print train.shape
print test.shape
test.to_csv(
    'data/test_data.csv',
    index=False
)
labels_train.to_csv(
    'data/train_labels.csv',
    index=False,
    header=True
)
train.to_csv(
    'data/train_data.csv',
    index=False
)
