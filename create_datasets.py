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
    'Neutered',
    'AnimalType',
    'AgeInYears'
    ]

dummies = [
    'isMix',
    'Breed_formatted',
    'Sex',
    'Neutered',
    'AnimalType'
    ]

predicted = 'OutcomeType'

labels_train = train[predicted]

train = train[features]
test = test[features]

train["Train"] = 1
test["Train"] = 0

combined = pd.concat([train, test])
combined_dummies = pd.get_dummies(combined, columns=dummies)

train = combined_dummies[combined_dummies["Train"] == 1]
test = combined_dummies[combined_dummies["Train"] == 0]
train = train.drop(["Train"], axis=1)
test = test.drop(["Train"], axis=1)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

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
