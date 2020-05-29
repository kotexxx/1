#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from tqdm import tqdm
import scipy.stats
from sklearn import datasets
from pandas import DataFrame
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from IPython.display import Image
#from graphviz import Digraph
from sklearn.externals.six import StringIO
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.tree import export_graphviz
from sklearn import tree
#from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.model_selection import KFold
from sklearn import preprocessing
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
# データマート作成
df = pd.read_csv("TrainingData20190909 2.csv",header=None)
#df = pd.read_csv("filter result.csv",header=None)
#df = df.astype(np.float32)
#df_scaled = preprocessing.scale(df)

#df_scaled.mean(axis=0)
#df_scaled.std(axis=0)
#a = df[df.iloc[:,17] <120]

#b = a[a.iloc[:,17] >= 90]


#sns.pairplot(df) 
#a = df[df.iloc[:,17] <120]

#b = a[a.iloc[:,17] >= 80]


#c = b[b.iloc[:,13] >= 15] 


# 説明変数、目的変数
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19]]#.values
y = df.iloc[:,[17]]#.values

#X = b.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
#y = b.iloc[:,[14]]#.values


#test data
do= pd.read_csv("TestingData20190909 2.csv",header=None)
#do = pd.read_csv("testt.csv",header=None)

#aa = do[do.iloc[:,17] <120]

#bb = aa[aa.iloc[:,17] >= 90]

#df_scaled = preprocessing.scale(bb)

#bb_scaled.mean(axis=0)
#bb_scaled.std(axis=0)


ds = do.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19]]#.values
dd = do.iloc[:,[17]]#.values,7,8,9,10,11,12,13,14,15

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


#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']


# In[2]:


# In[ ]:




#cross validation

#from sklearn.model_selection import train_test_split




# CSVファイルを読み込む
#df = pd.read_csv('weight.csv')
#df = pd.read_excel("Data11062019_intensity3.xlsx")
dff = pd.read_csv("Jtesting.csv")

#体重の選別
b = dff[dff.iloc[:,14] <= 110]
c = b[b.iloc[:,14] >= 85]
df = c[c.iloc[:,13] >= 10]

# 説明変数列と目的変数列に分割
label = df['O']
data  = df.drop('O', axis=1)

#print('label\n',label)
#print('data\n',data)
#print('-----')
    

# 学習用データとテストデータにわける
#train_data, test_data, train_label, test_label = train_test_split(data, label)

#kf = KFold(df, n_folds=5)
#分割数
kf = KFold(n_splits=5, shuffle=True)#, random_state=0)

x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
print(len(x))
print(x)
print(x[0],x[1],x[2],x[3],x[4])

j = 0
a = 0.00

for train,test in kf.split(df):
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    
    print('train\n',train_df)
    print('test\n',test_df)
    print('-----')

   # 学習する
    #clf = RandomForestRegressor()   
    #clf = RandomForestClassifier()  
    clf = xgb.XGBClassifier()        #94.6 
    #clf = xgb.XGBRegressor()        #95.46 #95.4 #95.44
    clf.fit(train_df.iloc[:,0:14],train_df.iloc[:,14])

    #test_label=test_df.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[4]]
    test_label=test_df.iloc[:,[14]]

    # 評価する
    predict = clf.predict(test_df.iloc[:,0:14])
    print(predict)
    rate_sum = 0

    #print("予測値"  "正解値")
    for i in range(len(test_df)):
        p = int(predict[i])
        t = int(test_label.iloc[i])
        rate_sum += int(min(t, p) / max(t, p) * 100)
        print(p, t)

    #print("精度")
    x[j] = rate_sum / len(test_label)
    print(rate_sum / len(test_label)) 
    print("x[j]", x[j])   
    j += 1
    
print(x[0],x[1],x[2],x[3],x[4])

for j in range(len(x)):
    print("x", x[j])
    a += x[j]
    j += 1

print("総合精度")
print(a/len(x))


# In[ ]:


# In[ ]:


_, axes = plt.subplots(1, 2)
xgb.plot_tree(bst, num_trees=2, ax=axes[0])
xgb.plot_tree(bst, num_trees=3, ax=axes[1])


# In[ ]:


# MSE

print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )
# R^2

print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )

# 出力
>>>print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )
MSE train : 1.687, test : 10.921
>>>print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
R^2 train : 0.981, test : 0.847
        
       
get_ipython().run_line_magic('matplotlib', 'inline')

# プロット
plt.figure(figsize = (10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', s = 35, alpha = 0.7, label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()

    



    
    
    #特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(y[indices[i]]) + "   " + str(feature[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), y[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()

xgb.to_graphviz(bst, num_trees=1)
_, axes = plt.subplots(1, 2)
xgb.plot_tree(bst, num_trees=2, ax=axes[0])
xgb.plot_tree(bst, num_trees=3, ax=axes[1])





ppln = Pipeline([

           ('scale', StandardScaler()),

           ('pca', PCA(0.80)),
           
         ('clf', mod)


       ])

train_sizes, train_scores, valid_scores = learning_curve(estimator=ppln,

                                              X=X, y=y,

                                              train_sizes=np.linspace(0.1, 1.0, 10),

                                              cv=5, n_jobs=1)
#学習曲線

# calculate the coorinates for plots

train_mean = np.mean(train_scores, axis=1)

train_std  = np.std(train_scores, axis=1)

valid_mean = np.mean(valid_scores, axis=1)

valid_std  = np.std(valid_scores, axis=1)





plt.style.use('seaborn-whitegrid')



# draw the training scores

plt.plot(train_sizes, train_mean, color='orange', marker='o', markersize=5, label='training accuracy')

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.1, color='orange')



# draw the validation scores

plt.plot(train_sizes, valid_mean, color='darkblue', marker='o', markersize=5,label='validation accuracy')

plt.fill_between(train_sizes, valid_mean + valid_std,valid_mean - valid_std, alpha=0.1, color='darkblue') 



plt.xlabel('#training samples')

plt.ylabel('accuracy')

plt.legend(loc='lower right')

plt.ylim([0.7, 1.01])

plt.show()





gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
classifier.fit(X, y)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)












def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
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
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.0001)
plot_learning_curve(estimator, title, X, y, (0, 1.01), cv=cv, n_jobs=4)

plt.show()

