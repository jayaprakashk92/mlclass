import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_excel("titanic3.xls")

df['ticket_id'] = pd.factorize(df.ticket)[0]
df['sex_id'] = pd.factorize(df.sex)[0]




mragefill = df[df.name.str.match(r'.*\bMr\b.*')].age.dropna().mean()
mrsagefill = df[df.name.str.match(r'.*\bMrs\b.*')].age.dropna().mean()
masteragefill = df[df.name.str.match(r'.*\bMaster\b.*')].age.dropna().mean()
missagefill = df[df.name.str.match(r'.*\bMiss\b.*')].age.dropna().mean()


df.loc[df.name.str.match(r'.*\bMr\b.*') & df.age.isnull(),'age']=mragefill
df.loc[df.name.str.match(r'.*\bMrs\b.*') & df.age.isnull(),'age']=mrsagefill
df.loc[df.name.str.match(r'.*\bMaster\b.*') & df.age.isnull(),'age']=masteragefill
df.loc[df.name.str.match(r'.*\bMiss\b.*') & df.age.isnull(),'age']=missagefill
df = df.fillna(0)
df['boat_id'] = pd.factorize(df.boat)[0]
df = df[['sex','ticket','boat','boat_id','ticket_id','sex_id','age','survived']]
# df = df[['sex','ticket','boat','boat_id','ticket_id','sex_id','age','sibsp','parch','survived']]
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
# print(len(test))
# print(len(train))
# print(train.head())

print(train.iloc[:,3:7].head())
print(train.iloc[:,7].head())


params = {'n_estimators': 400, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

print (y_test)
print (y_pred)
