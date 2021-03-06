
# coding: utf-8

from __future__ import print_function
import pandas as pd
import numpy as np
import re

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# Define functions:
# Strip /'s and add a column: isMix
def isMix(x):
    if x.lower().find('mix') > 0: return 'mix'
    if x.find('/') > 0: return 'pure'
    else: return 'mixUnknown'


def formatMixBreed(x):
    if x.lower().find('mix') > 0: return x.replace(' Mix', '')
    if x.find('/'): return x.split('/', 1)[0]
    else: return x


def getFrequentBreeds(train):
    vc = train['Breed_formatted'].value_counts()
    return vc[vc > 150].index.values


def scaleFeature(column):
    mins = np.min(column, axis=0)
    maxs = np.max(column, axis=0)
    return (column - mins) / (maxs - mins)


def formatBreedGroups(x, breeds):
    x = str(x)
    if x in breeds: return x
    else: return 'Exotic'


def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'genderUnknown'


def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'neuteredUnknown'


# Approximates months and weeks to years
def format_age(x):
    x = str(x)
    if x.find('week') > 0: return float(re.sub('\D', '', x)) / 52
    if x.find('month') > 0: return float(re.sub('\D', '', x)) / 12
    if x.find('day') > 0: return float(re.sub('\D', '', x)) / 365
    if x.find('year') > 0: return float(re.sub('\D', '', x))
    else: return 0


train['Sex'] = train.SexuponOutcome.apply(get_sex)
train['Neutered'] = train.SexuponOutcome.apply(get_neutered)
test['Sex'] = test.SexuponOutcome.apply(get_sex)
test['Neutered'] = test.SexuponOutcome.apply(get_neutered)
train['AgeInYears'] = train.AgeuponOutcome.apply(format_age)
test['AgeInYears'] = test.AgeuponOutcome.apply(format_age)
train['AgeInYears'] = scaleFeature(train['AgeInYears'])
test['AgeInYears'] = scaleFeature(test['AgeInYears'])
train['Breed_formatted'] = train.Breed.apply(formatMixBreed)
train['isMix'] = train.Breed.apply(isMix)
test['Breed_formatted'] = test.Breed.apply(formatMixBreed)
test['isMix'] = test.Breed.apply(isMix)
breeds = getFrequentBreeds(train)
train['Breed_formatted'] =\
    train.Breed_formatted.apply(formatBreedGroups, args = (breeds,))
test['Breed_formatted'] =\
    test.Breed_formatted.apply(formatBreedGroups, args = (breeds,))

train.to_csv('data/train_cleaned.csv', index=False)
test.to_csv('data/test_cleaned.csv', index=False)


