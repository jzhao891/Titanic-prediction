# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

data_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test.csv")
df=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_1.csv")
df_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test_1.csv").filter(regex='Survived|Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')
df=df.filter(regex='Survived|Age_sc|Fare_sc|Sex|Surname|Pclass|Family|Title_id')

####

from sklearn.ensemble import RandomForestClassifier
X = df[:df.shape[0]].values[:, 1::]
y = df[:df.shape[0]].values[:, 0]

X_test = df_test[:df_test.shape[0]].values[:, 1::]

random_forest = RandomForestClassifier(max_depth=5,oob_score=True, n_estimators=1000)
random_forest.fit(X, y)

Y_pred = random_forest.predict(X_test)

print random_forest.score(X, y)
submission = pd.DataFrame({
	    "PassengerId": data_test["PassengerId"],
	    "Survived": Y_pred.astype(int)
	})
submission.to_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/prediction_2.csv", index=False)

#########Feature importances with RandomForest

#from sklearn import important_features
data_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test.csv")
df=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_1.csv")
df=df.filter(regex='Survived|Age_sc|SibSp|Parch|Fare_sc|Sex|Pclass|Child|FamilySize|Family|Title_id')
df_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test_1.csv")
X=df[:df.shape[0]].values[:,1::]
y=df[:df.shape[0]].values[:,1]
features_list = df.columns.values[1::]

# Fit a random forest with (mostly) default parameters to determine feature importance
forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
forest.fit(X, y)
feature_importance = forest.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

# Get the indexes of all features over the importance threshold
important_idx = np.where(feature_importance)[0]

# Get the sorted indexes of important features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
print "\nFeatures sorted by importance (DESC):\n", important_idx[sorted_idx]

# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
plt.yticks(pos, important_idx[sorted_idx[:-1]])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

sorted_idx
feature_importance
forest.get_params()
df.filter(regex='Survived|Age_sc|SibSp|Parch|Fare_[0, 7.896]|Fare_[7.896, 14.454]|Fare_[14.454, 31.275]|Fare_[31.275, 512.329]|Sex|Pclass|Child|FamilySize|Family|Title_id')
