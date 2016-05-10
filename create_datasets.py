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

#    .join(train.AgeInYears)

#    .join(pd.get_dummies(train.Breed_formatted, prefix='is'))

labels_train = pd.get_dummies(train.OutcomeType)

features_test = pd.get_dummies(test.isMix, prefix='is')\
    .join(pd.get_dummies(test.Sex, prefix='is'))\
    .join(pd.get_dummies(test.Neutered, prefix='is'))

#    .join(test.AgeInYears)
#    .join(pd.get_dummies(test.Breed_formatted, prefix='is'))

features_test.to_csv(
    'data/test_data.csv',
    index=False
)
labels_train.to_csv(
    'data/train_labels.csv',
    index=False
)
features_train.to_csv(
    'data/train_data.csv',
    index=False
)
