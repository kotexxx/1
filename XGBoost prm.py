#!/usr/bin/env python
# coding: utf-8

# In[2]:


# パッケージのインポート
import numpy as np
import pandas as pd
import xgboost as xgb
import pydotplus
import matplotlib.pyplot as plt
import os
import pydot
import scipy.stats
import six
from tqdm import tqdm
from matplotlib.pylab import rcParams
from sklearn import preprocessing
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import KFold
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from xgboost import plot_importance
from matplotlib import cm
import seaborn as sns
#import pickle
#from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')


#%matplotlib inline
#rcParams['figure.figsize'] = 12, 4


# データマート作成
df = pd.read_csv("TrainingData20190909 2.csv",header=None)



a = df[df.iloc[:,17] < 120]

b = a[a.iloc[:,17] >= 90]


#c = b[b.iloc[:,13] >= 15]


# 説明変数、目的変数
X = df.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]]#.values
y = df.iloc[:,[17]]#.values

#test data
do= pd.read_csv("070719data test 2.csv",header=None) 

aa = do[do.iloc[:,17] < 120]

bb = aa[aa.iloc[:,17] >= 90]


XX = do.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]]#.values
yy = do.iloc[:,[17]]#.values

#前処理　標準化 分散1　平均0
#sci1 = scipy.stats.zscore(X)
#print(sci1)

#sci2 = scipy.stats.zscore(XX)
#print(sci2)

#sci11=pd.DataFrame(sci1)
#sci11.to_csv('sci1.csv',header=False,index=False)

#sci22=pd.DataFrame(sci2)
#sci22.to_csv('sci2.csv',header=False,index=False)

#前処理　正規化　最小値0　最大値1

#mm = preprocessing.MinMaxScaler()


#X_min_max = mm.fit_transform(X)



#XX_min_max = mm.fit_transform(XX)

X1=pd.DataFrame(X)
X1.to_csv('sci1',header=False,index=False)

XX1=pd.DataFrame(XX)
XX1.to_csv('sci2',header=False,index=False)



#ハイパーパラメータのチューニング

# 動かすパラメータを明示的に表示


#dtrain = xgb.DMatrix(X.as_matrix(), label=y.tolist())
dtrain = xgb.DMatrix(X, label=y)
objective = do.iloc[:,[17]]



fit_params = {
    'eval_metric': 'rmse'
    }


#nrounds
#ブースティングを行う回数(決定木の本数)を意味する
#クロスバリデーションを行うことで最適なブースティング回数を求めることができる
params= {#"learning_rate":list(np.arange(0.06, 0.2, 0.02)),
    
    "learning_rate":[0.05],
        "max_depth":[6],
         "subsample":[0.07],
   #min_samples_split ":[23],
    "gamma":[15],
    "n_estimators":[280],
    #"max_features":[sqrt]
         #"colsample_bytree": [1],
        "min_child_weight":[5]
          #"gamma": list(np.arange(0, 20, 5))#,params.fit(X,y, **fit_params) 
        # "objective": (reg:logistic)
           #"scale_pos_weight":[1]
         }

def GSfit(params):
    #print(params)
    regressor = xgb.XGBRegressor(n_estimators=280)
    #grid= GridSearchCV(regressor,params,cv=3, param_grid=params, scoring='neg_mean_squared_error',verbose=2)
    grid= GridSearchCV(regressor,params,cv=5, scoring='neg_mean_squared_error',verbose=0)
    grid.fit(X,y)
    return grid
#print(param_grid)
#Grid search
grid = GSfit(params)
grid_best_params = grid.best_params_
grid_scores_df = pd.DataFrame(grid.cv_results_)
#grid_scores_df.to_csv('../CSV/grid_scores_' + name_param + '_' + str(num_hue) + '.csv', index=False)early_stopping_rounds




#best n by CV
cv=xgb.cv(grid_best_params,dtrain,num_boost_round=400,nfold=3 
         )
n_best = cv[cv['test-rmse-mean'] == cv['test-rmse-mean'].min()]['test-rmse-mean'].index[0]
grid_best_params['n_estimators'] = n_best + 1
#pd.io.json.json_normalize(grid_best_params).to_csv('../CSV/param_best_part_' +'.csv', index=False)#num_boost_round=200, 

#利用可能なパラメータの表示
#Estimator.get_params（）

#fit by best params
regressor = xgb.XGBRegressor(learning_rate=grid_best_params['learning_rate'],
                             max_depth=grid_best_params['max_depth'],
                              subsample=grid_best_params['subsample'],
                             n_estimators=grid_best_params['n_estimators'],
                            #gamma=grid_best_params['gamma'],
                             #max_features=grid_best_params['max_features'],
                             # min_samples_split =grid_best_params['min_samples_split']
                           min_child_weight=grid_best_params['min_child_weight'],
                           gamma=grid_best_params['gamma']
                            )
regressor.fit(X, y,  verbose=False)

print(grid.best_params_)
print(grid.best_score_)

#save model
#pickle.dump(regressor, open('model.pkl', 'wb'))


# prediction
result = regressor.predict(XX)
#XX[objective] = yy
#XX[objective + '_pred'] = result

# Metric Use By Kaggle
def compute_rmsle(y_true, y_pred):
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true)
        
    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)
     
    return(np.average((np.log1p(y_pred) - np.log1p(y_true))**2)**.5)



print("RMSE: {0}".format(mean_squared_error(yy,result)**.5))
#比較グラフrmsとn_estimators

plt.rcParams["font.size"] = 16 #フォントサイズの指定
fig = plt.figure(figsize=(10,10)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid(True)

ax.scatter(cv.index, cv['train-rmse-mean']) #データのプロット
ax.scatter(cv.index, cv['test-rmse-mean']) #データのプロット
#ax.plot(x,x)
ax.set_xlim([None, None]) #x軸の範囲
ax.set_ylim([None, None]) #y軸の範囲

ax.set_title('name') #グラフのタイトル
#ax.set_xlabel(objective) #x軸の名前
#ax.set_ylabel(objective_pred) #y軸の名前

ax.legend(loc='upper left', fontsize=20) #凡例の表示
#plt.savefig('../Graph/' + 'name' +  '_'  + '_rmse-train-valid.png', bbox_inches="tight")
plt.show()

#結果を出力
df1=pd.DataFrame(result)
df1.to_csv('test4.csv',header=False,index=False)


xgb.plot_importance(regressor)







#拡大表示
plt.rcParams["font.size"] = 16 #フォントサイズの指定
fig = plt.figure(figsize=(10,10)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid(True)

#ax.scatter(cv.index, cv['train-rmse-mean']) #データのプロット
ax.scatter(cv.index, cv['test-rmse-mean']) #データのプロット
#ax.plot(x,x)
ax.set_xlim([40, 400]) #x軸の範囲
ax.set_ylim([3,7]) #y軸の範囲

ax.set_title('name') #グラフのタイトル
#ax.set_xlabel(objective) #x軸の名前
#ax.set_ylabel(objective_pred) #y軸の名前

ax.legend(loc='upper left', fontsize=20) #凡例の表示
#plt.savefig('../Graph/' + 'name' +  '_'  + 'rmse-train-valid-focus.png', bbox_inches="tight")
plt.show()





#過少予測と過大予測
#residuals = (XX['count_pred'] - XX['count'])

#plt.hist(residuals)
#plt.grid_search(True)
#plt.xlabel('(Predicted - Actual)')
#plt.ylabel('Count')
#plt.title('Residuals Distribution')
#plt.axvline(color='g')
#plt.show()








#inchild  heatmap
#aram_pivot = grid_scores_df.pivot_table(index=['param_gamma'], columns=['param_min_child_weight'], values=['mean_test_score'])*-1

#plt.rcParams["font.size"] = 12
#fig = plt.figure(figsize=(20,12)) #図の生成
#ax= fig.add_subplot(1,1,1) #軸の指定'Blues_r'

#sns.heatmap(param_pivot.loc[:,'mean_test_score'],
#            ax=ax, vmin=None, vmax=None, annot=True, fmt="1.3f", cmap='Blues_r')

#ax.set_label()

#ax.set_title(name + '_' + 'param') #グラフのタイトル
#ax.set_xlabel('min_child_weight') #x軸の名前
#ax.set_ylabel('gamma') #y軸の名前
 
#plt.savefig('../Graph/' + name + '_heatmap.png', bbox_inches="tight")
#plt.show() #グラフの表示


# In[12]:


params={'max_depth': [5],
        'subsample': [0.95],
        'colsample_bytree': [1.0]
}

print(params)

gs = GridSearchCV(mod,
                  params,
                  cv=10,
                  n_jobs=1,
                  verbose=2)
gs.fit(X,y)
predict2 = gs.predict(ds)

confusion_matrix(dd, predict2)


from bayes_opt import BayesianOptimization
def xgboost_cv(
    learning_rate,
    max_depth,
    subsample,
    colsample_bytree,
    min_child_weight,
    gamma,
    alpha):
    
    params = {}
    params['learning_rate'] = learning_rate
    # maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.
    params['max_depth'] = int(max_depth) 
    #  subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
    params['subsample'] = subsample
    # subsample ratio of columns when constructing each tree.
    params['colsample_bytree'] = colsample_bytree 
    # minimum sum of instance weight (hessian) needed in a child. The larger, the more conservative the algorithm will be.
    params['min_child_weight'] = min_child_weight
    # minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
    params['gamma'] = gamma 
    # L1 regularization term on weights, increase this value will make model more conservative. 
    params['alpha'] = alpha 
    params['objective'] = 'binary:logistic'

    cv_result = xgb.cv(
        params,
        xgtrain,
        num_boost_round=10, 
        nfold=3,
        seed=0,
        # Validation error needs to decrease at least every <stopping_rounds> round(s) to continue training.
        # callbacks=[xgb.callback.early_stop(20)]
    )

    return 1.0 - cv_result['test-error-mean'].values[-1]


xgboost_cv_bo = BayesianOptimization(xgboost_cv, 
                             {
                                 'learning_rate': (0.1, 0.9),
                                 'max_depth': (5, 15),
                                 'subsample': (0.5, 1),
                                 'colsample_bytree': (0.1, 1),
                                 'min_child_weight': (1, 20),
                                 'gamma': (0, 10),
                                 'alpha': (0, 10),
                             })

xgboost_cv_bo.maximize(n_iter=50)

 train_metric = evals_result['train']['rmse']
    plt.plot(train_metric, label='train rmse')
    eval_metric = evals_result['eval']['rmse']
    plt.plot(eval_metric, label='eval rmse')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('rmse')
    plt.show()


# In[3]:


# モデルにインスタンス生成
mod = xgb.XGBRegressor()
# ハイパーパラメータ探索

clf = GridSearchCV(mod, params, cv = 10, verbose=1, n_jobs =-1)
clf.fit(X,y)


predict = clf.predict(ds)

#print(clf.cv_results_, clf.best_params_, clf.best_score_)


#print(cv)

# 学習用、検証用データ作成

#(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3)#,random_state = 0


#print(dd)
# モデルのインスタンス作成
#mod = xgb.XGBRegressor()

#mod.fit(X_train, y_train)

#print(cv.best_params_, cv.best_score_)
#X_predict= cv.predict(ds)
#Y_predict= cv.predict(dd)
#predict = mod.predict(X_test)
#y_train_pred = mod.predict(ds)
#y_test_pred = mod.predict(X_test)
#mods = mod.fit(X,y)


def compute_rmsle(y_true, y_pred):
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true)
        
    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)
     
    return(np.average((np.log1p(y_pred) - np.log1p(y_true))**2)**.5)
plt.rcParams["font.size"] = 16 #フォントサイズの指定
fig = plt.figure(figsize=(10,10)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid(True)

ax.scatter(cv.index, cv['train-rmse-mean']) #データのプロット
ax.scatter(cv.index, cv['test-rmse-mean']) #データのプロット
ax.set_xlim([None, None]) #x軸の範囲
ax.set_ylim([None, None]) #y軸の範囲

ax.set_title('name') #グラフのタイトル
#ax.set_xlabel(objective) #x軸の名前
#ax.set_ylabel(objective_pred) #y軸の名前

ax.legend(loc='upper left', fontsize=20) #凡例の表示
plt.savefig('../Graph/' + 'name' +  '_'  + '_rmse-train-valid.png', bbox_inches="tight")
plt.show()
    



   # xgb.to_graphviz(clf, num_trees=1)


# In[ ]:


# 動かすパラメータを明示的に表示
params = {"eta":[0.05,0.08,0.1,0.3],
        "max_depth": [4,5,6,7,8,9,10],
         "subsample":[0.5,0.8,0.9,1],
         "colsample_bytree": [0.5,0.6,0.8,1.0],
          "min_child_weight":[0.5,15,300],
         }


# モデルにインスタンス生成
mod = xgb.XGBRegressor()
# ハイパーパラメータ探索
cv = GridSearchCV(mod, params, cv = 10,  n_jobs =-1)


print(cv)


mean_list = []

for s in clf.cv_results_:
    rms = sqrt(mean_squared_error(dd,predict))
    mean_list.append(s[1])#.append
    print(rms)
    

plt.figure(figsize=(4,2))
plt.plot(list(range(1,130,5)),mean_list )
plt.title("GridSearchCV Score")
plt.xlabel("n_estimators")
plt.show()
clf.cv_results_, clf.best_params_

mean_list = []

for s in clf.cv_results_:
    rms = sqrt(mean_squared_error(dd,predict))
    mean_list.append(s[1])#.append
    print(rms)
    

plt.figure(figsize=(4,2))
plt.plot(list(range(1,130,5)),mean_list )
plt.title("GridSearchCV Score")
plt.xlabel("n_estimators")
plt.show()
clf.cv_results_, clf.best_params_



#print(predict)

rms = sqrt(mean_squared_error(dd,predict))
print(rms)

plt.ylabel("Predict")
plt.xlabel("Actual")
plt.scatter(dd, predict)
plt.plot([60, 130], [60, 130], c='r')
plt.show()

for X,dd in kf.split(df):
    
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    
    print('train\n',train_df)
    print('test\n',test_df)
    print('-----')

    
    
    
    
    
    
    
    
    grid= {
   'eval_metric': 'rmse',
    'eval_set': [[X,y]]
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
plt.rcParams["font.size"] = 16 #フォントサイズの指定
fig = plt.figure(figsize=(10,10)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid_search(True)

ax.scatter(cv.index, cv['train-rmse-mean']) #データのプロット
ax.scatter(cv.index, cv['test-rmse-mean']) #データのプロット
#ax.plot(x,x)
ax.set_xlim([None, None]) #x軸の範囲
ax.set_ylim([None, None]) #y軸の範囲

ax.set_title('name') #グラフのタイトル
#ax.set_xlabel(objective) #x軸の名前
#ax.set_ylabel(objective_pred) #y軸の名前

ax.legend(loc='upper left', fontsize=20) #凡例の表示
plt.savefig('../Graph/' + 'name' +  '_'  + '_rmse-train-valid.png', bbox_inches="tight")
plt.show()

xgb.plot_importance(regressor)


#過少予測と過大予測
#residuals = (XX['count_pred'] - XX['count'])

#plt.hist(residuals)
#plt.grid_search(True)
#plt.xlabel('(Predicted - Actual)')
#plt.ylabel('Count')
#plt.title('Residuals Distribution')
#plt.axvline(color='g')
#plt.show()


plt.rcParams["font.size"] = 12 #フォントサイズの指定
fig = plt.figure(figsize=(6,6)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid(True)

ax.scatter(XX[objective], XX[objective + '_pred']) #データのプロット
x = np.arange(0, 10, 0.1)
ax.plot(x,x, color='red')
ax.set_xlim([0, 8]) #x軸の範囲
ax.set_ylim([0, 8]) #y軸の範囲


ax.set_title('name') #グラフのタイトル
ax.set_xlabel(objective) #x軸の名前
ax.set_ylabel(objective + '_pred') #y軸の名前

ax.legend(loc='upper left') #凡例の表示
plt.savefig('../Graph/' + 'name' +  '_'  + '_scatter.png', bbox_inches="tight")
#plt.show()


#パラメータとvalidation errorの関係
plt.rcParams["font.size"] = 12
fig = plt.figure(figsize=(12,8)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

sns.heatmap(param_pivot.loc[:,'mean_test_score'],
            ax=ax, vmin=0.085, vmax=0.088, annot=True, fmt="1.3f", cmap='Blues_r')

#ax.set_label()
ax.set_title(name + '_' + 'param') #グラフのタイトル
ax.set_xlabel('learning_rate') #x軸の名前
ax.set_ylabel('max_depth') #y軸の名前
 
plt.savefig('../Graph/' + name + '_heatmap.png', bbox_inches="tight")
plt.show() #グラフの表示



ax.scatter(XX(yy), XX(yy+ '_pred')) #データのプロット
x = np.arange(0, 10, 0.1)
ax.plot(x,x, color='red')
ax.set_xlim([0, 8]) #x軸の範囲
ax.set_ylim([0, 8]) #y軸の範囲


ax.set_title('name') #グラフのタイトル
ax.set_xlabel(objective) #x軸の名前
ax.set_ylabel(objective + '_pred') #y軸の名前

ax.legend(loc='upper left') #凡例の表示
plt.savefig('../Graph/' + 'name' +  '_'  + '_scatter.png', bbox_inches="tight")
#plt.show()

#パラメータとvalidation errorの関係
plt.rcParams["font.size"] = 12
fig = plt.figure(figsize=(12,8)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定


max_depth = grid_scores_df['param_max_depth'].ravel()
learning_rate = grid_scores_df['param_learning_rate'].ravel()
rmse_valid = grid_scores_df['mean_test_score'].ravel() * -1

param_pivot = grid_scores_df.pivot_table(index=['param_max_depth'], columns=['param_learning_rate'], values=['mean_test_score']) * -1
sns.heatmap(param_pivot.loc[:,'mean_test_score'],
            ax=ax, vmin=0.085, vmax=0.088, annot=True, fmt="1.3f", cmap='Blues_r')

#best n by CV
cv=xgb.cv(grid_best_params,dtrain,num_boost_round=400,nfold=5)
n_best = cv[cv['test-rmse-mean'] == cv['test-rmse-mean'].min()]['test-rmse-mean'].index[0]
grid_best_params['n_estimators'] = n_best + 1




plt.rcParams["font.size"] = 16 #フォントサイズの指定
fig = plt.figure(figsize=(10,10)) #図の生成
ax = fig.add_subplot(1,1,1) #軸の指定

plt.grid(True)

ax.scatter(cv.index, cv['train-rmse-mean']) #データのプロット
ax.scatter(cv.index, cv['test-rmse-mean']) #データのプロット
#ax.plot(x,x)
ax.set_xlim([None, None]) #x軸の範囲
ax.set_ylim([None, None]) #y軸の範囲

ax.set_title('name') #グラフのタイトル
#ax.set_xlabel(objective) #x軸の名前
#ax.set_ylabel(objective_pred) #y軸の名前

ax.legend(loc='upper left', fontsize=20) #凡例の表示
#plt.savefig('../Graph/' + 'name' +  '_'  + '_rmse-train-valid.png', bbox_inches="tight")
plt.show()

