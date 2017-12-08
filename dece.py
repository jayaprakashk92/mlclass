
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import ensemble
from sklearn import datasets
from sklearn import tree
import graphviz
#sns.set(style="ticks")
df=pd.read_excel('/home/indixuser/Downloads/titanic3.xlsx')
df['boat']=df[['boat']].fillna(999)



df = df.fillna(0)
df['pname'] = pd.factorize(df.name)[0]
df['sex_id'] = pd.factorize(df.sex)[0]

df['boat_id'] = pd.factorize(df.boat)[0]



train=df[['name','pclass','pname','sex_id','age','boat_id','survived']]
#train=train.loc[0:500,:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(df[['pclass','pname','sex_id','age','boat_id']],df[['survived']])

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("titanic")


