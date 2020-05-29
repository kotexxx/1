#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Import required libraries

import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')

#traning data
df = pd.read_csv("Jtesting.csv",header=None)
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
y = df.iloc[:,[14]]#.values
#test data
do= pd.read_csv("testt.csv",header=None) 

#aa = do[do.iloc[:,14] <= 120]

#bb = aa[aa.iloc[:,14] >= 80]


XX = do.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]#.values
yy = do.iloc[:,[14]]#.values

#LightGBM
def run_lgb(X,y,XX, yy):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.004,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(X, label=y)
    lgval = lgb.Dataset(XX, label=yy)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result

# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")
# Training XGB
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")


#Catboost
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)

cb_model.fit(dev_X, dev_y,
             eval_set=(val_X, val_y),
             use_best_model=True,
             verbose=True)

pred_test_cat = np.expm1(cb_model.predict(X_test))

#Combine Predictions

sub = pd.read_csv('../input/sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test

sub_xgb = pd.DataFrame()
sub_xgb["target"] = pred_test_xgb

sub_cat = pd.DataFrame()
sub_cat["target"] = pred_test_cat

sub["target"] = (sub_lgb["target"] * 0.5 + sub_xgb["target"] * 0.3 + sub_cat["target"] * 0.2)


print(sub.head())
sub.to_csv('sub_lgb_xgb_cat.csv', index=False)


# In[ ]:




