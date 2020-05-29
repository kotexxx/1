#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBRegressor

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, 
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

df = pd.read_csv("Jtesting.csv",header=None)

X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
y = df.iloc[:,[14]]#.values


#do= pd.read_csv("TestingData20190909 2.csv",header=None)

#ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
#dd = bb.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15


xgb = XGBRegressor(min_child_weight =1,random_state=5, n_jobs=1)
std = StandardScaler()
x = std.fit_transform(X)


title = xgb
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
plt = plot_learning_curve(xgb, title, x, y, cv=3, ylim=(0.0, 1.01), n_jobs=1)
plt.show()


# In[34]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, 
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

df = pd.read_csv("TrainingData20190909 2.csv",header=None)

X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
y = df.iloc[:,[17]]#.values


#do= pd.read_csv("TestingData20190909 2.csv",header=None)

#ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
#dd = bb.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15


RandomForest = RandomForestRegressor(max_depth = 5,max_features=16,random_state=5, n_jobs=1)
std = StandardScaler()
x = std.fit_transform(X)


title = RandomForest
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
plt = plot_learning_curve(RandomForest, title, x, y, cv=10, ylim=(0.0, 1.01), n_jobs=1)
plt.show()


# In[ ]:




