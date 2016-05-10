from modelfunctions import *
import pandas as pd
from sklearn.cross_validation import train_test_split
import pprint
import evaluation
from sklearn.metrics import classification_report, accuracy_score
pp = pprint.PrettyPrinter(indent=4)


X_test = pd.read_csv('data/test_data.csv')
Y = pd.read_csv('data/train_labels.csv')
X = pd.read_csv('data/train_data.csv')

print('Creating training and validation sets')
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state = 42
)

X_train, X_validate, y_train, y_validate = train_test_split(
    X_train, y_train, test_size = 0.2, random_state = 42
)

pred, res, best_fit = rfclassify(X_train, y_train, X_validate, y_validate, X_test)

#pred2, res2 = adaclassify(X_train, y_train, X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, res))
print(accuracy_score(y_test, res))


submission(best_fit, X, Y, X_test)
print('RF logloss:')
#print(evaluation.logloss(y_test, pred))
print('Adaboost logloss:')
#print(evaluation.logloss(y_test, pred2))
'''
np.save('pred-results', pred)
np.save('test-results', res)
print(X_test.shape)
print(X_test.shape)
print('Finally, let\'s create the output set')
#pred_final = fit.predict_proba(X_test)
#np.savetxt('output.csv', pred_final, delimiter=',')  
'''