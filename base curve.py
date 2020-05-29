#!/usr/bin/env python
# coding: utf-8

# In[5]:


# パッケージのインポート
import numpy as np
import pandas as pd
import xgboost as xgb
import pydotplus
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pydot
import six
import xlrd
import scipy.stats
from sklearn import datasets
from pandas import DataFrame
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from IPython.display import Image
#from graphviz import Digraph
from sklearn.tree import export_graphviz
from sklearn import tree
#from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.model_selection import KFold
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve



# データマート作成
df = pd.read_csv("TrainingData20190909 2.csv",header=None)
#df = pd.read_csv("filter result.csv",header=None)
#df = df.astype(np.float32)
#df_scaled = preprocessing.scale(df)

#df_scaled.mean(axis=0)
#df_scaled.std(axis=0)
a = df[df.iloc[:,17] <120]

b = a[a.iloc[:,17] >= 90]


#sns.pairplot(df) 
#a = df[df.iloc[:,17] <120]

#b = a[a.iloc[:,17] >= 80]


#c = b[b.iloc[:,13] >= 15] 


# 説明変数、目的変数
X = b.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
y = b.iloc[:,[17]]#.values

#X = b.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
#y = b.iloc[:,[14]]#.values


#test data
do= pd.read_csv("TestingData20190909 2.csv",header=None)
#do = pd.read_csv("testt.csv",header=None)

aa = do[do.iloc[:,17] <120]

bb = aa[aa.iloc[:,17] >= 90]

#df_scaled = preprocessing.scale(bb)

#bb_scaled.mean(axis=0)
#bb_scaled.std(axis=0)


ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
dd = bb.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15

#ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
#dd = bb.iloc[:,[14]]#.values



#前処理　標準化

#sci1 = scipy.stats.zscore(X)
#print(sci1)

#sci2 = scipy.stats.zscore(ds)
#print(sci2)

#sci11=pd.DataFrame(sci1)
#sci11.to_csv('sci1.csv',header=False,index=False)

#sci22=pd.DataFrame(sci2)
#sci22.to_csv('sci2.csv',header=False,index=False)


#前処理　　置換
# A列でaaという値を含むなら、そのセルの値をbbに置換する
#df.loc[df['A'].str.contains('aa'), 'A'] = 'bb'







# 学習用、検証用データ作成

#(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3)#,random_state = 0


#print(dd)
# モデルのインスタンス作成
mod = xgb.XGBRegressor()
#mod.fit(X_train, y_train)
mod.fit(X,y)
#X_predict= mod.predict(ds)
#Y_predict= mod.predict(dd)
#predict = mod.predict(X_test)
#y_train_pred = mod.predict(ds)
#y_test_pred = mod.predict(X_test)
mods = mod.fit(X,y)
predict = mods.predict(ds)

#print(predict)
#print(predict)


rms = np.sqrt(mean_squared_error(dd,predict))
print(rms)

#df1=pd.DataFrame(predict)
#df1.to_csv('test.csv',header=False,index=False)

#df2=pd.DataFrame(dd)
#df2.to_csv('test2.csv',header=False,index=False)


rate_sum = 0


#print("予測値 "   "正解値")
#for i in range(len(dd)): # for i in range(len(y_test)):
#  p = int(predict[i])
#  t = int(dd.iloc[i])  #t = int(y_test.iloc[i]) 
#  rate_sum += int(min(t, p) / max(t, p))#minとmaxでtかpを選択して大きいほうで割ることで1以下の数値になる(実測値と推測値を比べている)
#  print(p , t)

#print("精度 ")
#print(rate_sum / len(dd))#平均値を算出len y_test

evals_result = {} 

#特徴量の重要度
feature = mods.feature_importances_


def Snippet_188(): 
    print()
    print(format('Hoe to evaluate XGBoost model with learning curves','*^82'))    
    
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    # load libraries
    import numpy as np
    import xgboost as xgb
    import matplotlib.pyplot as plt    
    
    plt.style.use('ggplot')
    
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    df = pd.read_csv("TrainingData20190909 2.csv",header=None)
    X = b.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
    y = b.iloc[:,[17]]#.values

    do= pd.read_csv("TestingData20190909 2.csv",header=None)


    ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
    dd = bb.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15
    mod = xgb.XGBRegressor()
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(mod, 
                                               X, y, cv=10, scoring='accuracy', n_jobs=-1, 
                                               # 50 different sizes of the training set
                                               train_sizes=np.linspace(0.01, 1.0, 50))
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Draw lines
    plt.subplots(figsize=(12,12))
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout(); plt.show()
Snippet_188()




def Snippet_189():
    print()
    print(format('Hoe to visualise XGBoost model with learning curves','*^82'))
    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    from numpy import loadtxt
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from matplotlib import pyplot
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    # load data
    df = pd.read_csv("TrainingData20190909 2.csv",header=None)



    # 説明変数、目的変数
    X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
    y = df.iloc[:,[17]]#.values


    #test data
    do= pd.read_csv("TestingData20190909 2.csv",header=None)
    #do = pd.read_csv("testt.csv",header=None)

    ds = do.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]#.values
    dd = do.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15


    # fit model no training data
    mod = xgb.XGBRegressor()
    eval_set = [(X, y), (ds, dd)]
    mod.fit(X, y, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    # make predictions for test data
    y_pred = mod.predict(ds)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(dd, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # retrieve performance metrics
    results = mod.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = pyplot.subplots(figsize=(15,15))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()

    # plot classification error
    fig, ax = pyplot.subplots(figsize=(15,15))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Error')
    pyplot.title('XGBoost  Error')
    pyplot.show()

Snippet_189()






plt.figure(figsize = (10, 7))
plt.scatter(ds, ds - X, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'Training data')
plt.scatter(dd, dd - y, c = 'lightgreen', marker = 's', s = 35, alpha = 0.7, label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()


# In[ ]:




