%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

from sklearn import ensemble
from sklearn import datasets

boston = datasets.load_boston()
#print (boston)

X_train, y_train = X[:50], y[:50]
X_test, y_test = X[51:100], y[51:100]


params = {'n_estimators': 400, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

print (y_test)
print (y_pred)
