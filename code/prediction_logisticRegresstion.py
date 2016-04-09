# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

#############start logistic regression

df=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_1.csv")
data_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test.csv")
#df=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_2.csv")
#df=df.filter(regex='Survived|Age_sc|SibSp|Parch|Fare_[0, 7.896]|Fare_sc|Sex|Pclass|Child|FamilySize|Family|Title_id')
df_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test_1.csv")
train_df = df.filter(regex='Survived|Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')
train_np = train_df.as_matrix()
train_np

# the result of Survival
y = train_np[:, 0]

# X is the set of feature
X = train_np[:, 1:]

# fit those feature with BaggingRegressor
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf
bagging_clf = BaggingRegressor(clf, n_estimators=200, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')
test.info()
predictions = bagging_clf.predict(test)

predictions.astype(np.int32)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/prediction_2.csv", index=False)
########cross-validation(do it over and over)

#score model by using cross-validation
print cross_validation.cross_val_score(clf,X,y,cv=5)

#slice 70% training data from the whole as cross-validation training data,and 30% training data for testing
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_cv = split_train.filter(regex='Survived|Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')

#model logistic regression with splited training data
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_cv.as_matrix()[:,1:], train_cv.as_matrix()[:,0])

#test model with splited training data(split_cv)
test_cv = split_cv.filter(regex='Survived|Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')
predictions = clf.predict(test_cv.as_matrix()[:,1:])

#find out bad case for further analysis and optimization
origin_data_train = pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_1.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != test_cv.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases

