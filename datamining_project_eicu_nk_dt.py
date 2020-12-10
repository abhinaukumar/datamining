# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:04:11 2020

@author: nisha
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
#%%#
os.chdir("C:/Users/Acer/Desktop/Nisha/Nisha Career/PhD/Coursework/Fall 2020/Project/eICU/Data/")
#%%#
eicu = pd.read_csv('eicu_features.csv.gz', compression='gzip')
#test=eicu.head(150000)
eicu.drop(columns=['Unnamed: 0', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
eicu.set_index('patientunitstayid', inplace=True)
X=eicu.drop(['rlos'],axis=1)
y=eicu['rlos']
stayids = X.index.unique()

#%%# Train test split
train_ids, test_ids = train_test_split(stayids, test_size=0.2, random_state=0)
n_trn_ids = len(train_ids)
n_tst_ids = len(test_ids)

X_train, X_test = X.loc[train_ids], X.loc[test_ids]
y_train, y_test = y.loc[train_ids], y.loc[test_ids]

#%%# Model preprocessing functions and classes

# Declare TargetEncoder class for categorical variables
class TargetEncoder():
    def __init__(self):
        self.category_maps = {}
        return

    def keys(self):
        return self.category_maps.keys()

    def fit(self, X, y, keys):
        if type(keys) != list:
            keys = [keys]

        for key in keys:
            print("Fitting column {}".format(key))
            category_map = {}
            for category, group in X.groupby(key, as_index=False):
                category_map[category] = y.loc[y.index.isin(group.index)].mean()
            category_map[''] = y.mean()
            self.category_maps[key] = category_map

    def transform(self, X):
        retX = X.copy()
        for key in retX.keys():
            if key in self.category_maps:
                retX[key] = retX[key].map(lambda x: self.category_maps[key][x] if x in self.category_maps[key] else self.category_maps[key][''])
        
        return retX

def input_scaling(data):
       print('Fitting MinMaxScaler...')
       scaler = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(X_trn)
       X_temp = scaler.transform(data.values)
       X_transformed = pd.DataFrame(X_temp, index=data.index, columns=data.columns)
       return X_transformed


def compute_metrics(y_true, y_pred):
       mae=metrics.mean_absolute_error(y_true, y_pred)
       mse = metrics.mean_squared_error(y_true, y_pred)
       rmse=np.sqrt(metrics.mean_squared_error(y_true, y_pred))
       r2 = r2_score(y_true, y_pred)
       return [mae, mse, rmse, r2]
#%%#
#Target encoding the inputs and test data transforms
encoder=TargetEncoder()
encoder.fit(X,y,['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
X_trn=encoder.transform(X_train)
X_tst=encoder.transform(X_test)

#Scaling the inputs and test data transforms
X_trn_scld=input_scaling(X_trn)
X_tst_scld=input_scaling(X_tst)

#%%#  
# create a Vanilla decisontree regressor object 
regressor1 = DecisionTreeRegressor(random_state = 0, max_depth=5,  min_samples_split=10, min_samples_leaf=100)  
  
# fit the regressor with X and Y data 
regressor1.fit(X_trn_scld, y_train) 
y_pred = regressor1.predict(X_tst_scld) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df['Actual'].mean()

mae, mse, rmse, r2=compute_metrics(y_test,y_pred)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('No. of leaves', regressor1.get_n_leaves())

regressor = DecisionTreeRegressor(random_state = 0, max_depth=5,  min_samples_split=10, min_samples_leaf=100)  
 
test=X_trn_scld.head() 
# fit the regressor with X_train and Y_train datadirectly without encoding and scaling
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df['Actual'].mean()

mae, mse, rmse, r2=compute_metrics(y_test,y_pred)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('No. of leaves', regressor.get_n_leaves())
#%%#
# Doing a gridsearchCV and 5-fold and 10 fold CV R sq upto 0.08
scoring = make_scorer(r2_score)
param_grid={"max_depth": [10, 15],
              "max_leaf_nodes": [20, 100],
              "min_samples_leaf": [20, 40, 100],
              'min_samples_split': [10, 20, 40]}
g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, scoring=scoring, cv=5, refit=True)
g_cv.fit(X_trn_scld, y_train)
g_cv.best_params_

result = g_cv.cv_results_
# print(result)
r2_score(y_test, g_cv.best_estimator_.predict(X_tst_scld))
mae, mse, rmse, r2=compute_metrics(y_test,g_cv.best_estimator_.predict(X_tst_scld))
print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('No. of leaves', regressor.get_n_leaves())
#%%#
#Predicting just LOS i.e., 1st record for each patient id.

LOS_eicu=eicu.groupby('patientunitstayid').first()
test=LOS_eicu.head()
#Some patient unit stay ids do not start with offset 1
eicu.loc[141631.0]

#
X=LOS_eicu.drop(['rlos','offset'],axis=1)
y=LOS_eicu['rlos']
stayids = X.index.unique()

# Train test split
train_ids, test_ids = train_test_split(stayids, test_size=0.2, random_state=0)
n_trn_ids = len(train_ids)
n_tst_ids = len(test_ids)

X_train, X_test = X.loc[train_ids], X.loc[test_ids]
y_train, y_test = y.loc[train_ids], y.loc[test_ids]

#Target encoding the inputs and test data transforms
encoder=TargetEncoder()
encoder.fit(X,y,['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
X_trn=encoder.transform(X_train)
X_tst=encoder.transform(X_test)

#Scaling the inputs and test data transforms
X_trn_scld=input_scaling(X_trn)
X_tst_scld=input_scaling(X_tst)

# 
# create a Vanilla decisontree regressor object 
regressor = DecisionTreeRegressor(random_state = 0, max_depth=5,  min_samples_split=10, min_samples_leaf=100)  
  
# fit the regressor with X and Y data 
regressor.fit(X_trn_scld, y_train) 
y_pred = regressor.predict(X_tst_scld) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df['Actual'].mean()

mae, mse, rmse, r2=compute_metrics(y_test,y_pred)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('No. of leaves', regressor.get_n_leaves())

# Doing a gridsearchCV and 5-fold and 10 fold CV R sq upto 0.08
scoring = make_scorer(r2_score)
param_grid={"max_depth": [10, 15, 20],
              "max_leaf_nodes": [20, 100, 200],
              "min_samples_leaf": [100, 500, 1000],
              'min_samples_split': [10, 20, 40, 100]}
g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, scoring=scoring, cv=5, refit=True)
g_cv.fit(X_trn_scld, y_train)
g_cv.best_params_

result = g_cv.cv_results_
# print(result)
r2_score(y_test, g_cv.best_estimator_.predict(X_tst_scld))
mae, mse, rmse, r2=compute_metrics(y_test,g_cv.best_estimator_.predict(X_tst_scld))
print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
#%%#
#Predicting just LOS i.e., 1st record for each patient id with an additional feature of number of offsets.
test=eicu.head(100)
pat_offsets=eicu.groupby('patientunitstayid')['offset'].size().reset_index()
pat_offsets.set_index('patientunitstayid', inplace=True)
pat_offsets.index

rlos_eicu=LOS_eicu.drop(['offset'],axis=1)
rlos_eicu.index
rlos_eicu=rlos_eicu.join(pat_offsets, how='inner')

X=rlos_eicu.drop(['rlos'],axis=1)
y=rlos_eicu['rlos']
stayids = X.index.unique()

# Train test split
train_ids, test_ids = train_test_split(stayids, test_size=0.2, random_state=0)
n_trn_ids = len(train_ids)
n_tst_ids = len(test_ids)

X_train, X_test = X.loc[train_ids], X.loc[test_ids]
y_train, y_test = y.loc[train_ids], y.loc[test_ids]

#Target encoding the inputs and test data transforms
encoder=TargetEncoder()
encoder.fit(X,y,['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
X_trn=encoder.transform(X_train)
X_tst=encoder.transform(X_test)

#Scaling the inputs and test data transforms
X_trn_scld=input_scaling(X_trn)
X_tst_scld=input_scaling(X_tst)

# 
# create a Vanilla decisontree regressor object 
regressor = DecisionTreeRegressor(random_state = 0, max_depth=5,  min_samples_split=10, min_samples_leaf=100)  
  
# fit the regressor with X and Y data 
regressor.fit(X_trn_scld, y_train) 
y_pred = regressor.predict(X_tst_scld) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df['Actual'].mean()

mae, mse, rmse, r2=compute_metrics(y_test,y_pred)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
print('No. of leaves', regressor.get_n_leaves())

# Doing a gridsearchCV and 5-fold and 10 fold CV R sq upto 0.08
scoring = make_scorer(r2_score)
param_grid={"max_depth": [10, 15, 20], 
            "max_leaf_nodes": [20, 100, 200],
            "min_samples_leaf": [100, 500, 1000],
            'min_samples_split': [10, 20, 40, 100]
              }
g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, scoring=scoring, cv=5, refit=True)
g_cv.fit(X_trn_scld, y_train)
g_cv.best_params_

result = g_cv.cv_results_
# print(result)
r2_score(y_test, g_cv.best_estimator_.predict(X_tst_scld))
mae, mse, rmse, r2=compute_metrics(y_test,g_cv.best_estimator_.predict(X_tst_scld))
print('Mean Absolute Error:', mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
#%%#
# Descriptives  for presentation.
rlos_eicu['rlos'].mean()
rlos_eicu['rlos'].median()
rlos_eicu['rlos'].max()
rlos_eicu['rlos'].min()

dataTypeSeries = rlos_eicu.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)

rlos_eicu.groupby('gender')['gender'].count()
cont_feat=rlos_eicu.drop(['Eyes','GCS Total','Motor','Verbal','apacheadmissiondx'],axis=1)
corr=cont_feat.corr()
sns.set(rc={'figure.figsize':(19,15)},font_scale=0.8)
sns.heatmap(corr, center=0, xticklabels=corr.columns, yticklabels=corr.columns,cmap='coolwarm')


ax=rlos_eicu.boxplot()
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
rlos_eicu.hist()

from pandas.plotting import scatter_matrix
scatter_matrix(rlos_eicu, alpha=0.2, figsize=(11, 7), diagonal='kde')

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydot

features=list(X_trn_scld.columns[0:])
dot_data=StringIO()
export_graphviz(g_cv.best_estimator_,out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph=pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

feat_imp=g_cv.best_estimator_.feature_importances_
FI=pd.DataFrame(feat_imp, columns=['vals'])
FI['Features']=features
FI.head()
temp=FI.sort_values(by='vals',ascending=False).reset_index()

sns.catplot(x="vals", y="Features", kind="bar", data=temp,palette=["C0", "C0", "C0"])
