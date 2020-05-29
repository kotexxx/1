#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pydotplus
import csv
import matplotlib.pyplot as plt
import os
import pydot
#import six
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from IPython.display import Image
from graphviz import Digraph
from sklearn.externals.six import StringIO
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.tree import export_graphviz
from sklearn import tree
#from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# CSVファイルを読み込む
#df = pd.read_csv('weight_2.csv')
df = pd.read_csv("Traindatafor05072019.csv",header=None)



#a = df[df.iloc[:,16] <120]

#b = a[a.iloc[:,16] >= 80]


#c = b[b.iloc[:,13] >= 15]


# 説明変数、目的変数
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]#.values
y = df.iloc[:,[16]]#.values

#test data
do= pd.read_csv("070719data test.csv",header=None)

aa = do[do.iloc[:,16] <120]

bb = aa[aa.iloc[:,16] >= 80]


ds = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]#.values
dd = bb.iloc[:,[16]]#.values,7,8,9,10,11,12,13,14,15


# 学習する
clf = RandomForestRegressor()
#clf = RandomForestClassifier()
clf = clf.fit(X, y)

"""
・n_estimators　→デフォルト値１０。決定木の数。
・max_features　→デフォルト値は特徴量数の平方根。決定木の特徴量の数。大きいと似たような決定木が増える、小さいと決定木がばらけるがデータに適合できにくくなる。
・random_state　→デフォルト値なし。乱数ジェネレータをどの数値から始めるか。前回と異なるモデルを得たい場合は数値を変更すると良いです。
max_depth　→デフォルト値なし。決定木の最大の深さ。
min_samples_split　→デフォルト値2。木を分割する際のサンプル数の最小数
"""
# 評価する  
predict = clf.predict(ds)

rate_sum = 0


print("予測値 "   "正解値")
for i in range(len(dd)):
  p = int(predict[i])
  t = int(dd.iloc[i]) 
  rate_sum += int(min(t, p) / max(t, p) * 100)#minとmaxでtかpを選択して大きいほうで割ることで1以下の数値になる(実測値と推測値を比べている)
  print(p , t)

print("精度 ")
print(rate_sum / len(dd))#平均値を算出
rms = np.sqrt(mean_squared_error(predict,dd))
print(rms)

#特徴量の重要度
feature = clf.feature_importances_

#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']

#特徴量の名前
label = df.columns[0:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()


#決定木の可視化
dot_data = StringIO()
i_tree=0

col=['weight','K length','K Perimeter length','K Radius','Total pixels','curvature','Tilt','K area','3D K length','3D weight','3D K length','3D K Perimeter length','3D K Radius','Height in 3D','Average height in 3D','Luminance value']
for tree_in_forest in clf.estimators_:
    export_graphviz(tree_in_forest, out_file='tree dot',feature_names=col, max_depth=3)#max_depth=5
    (graph,)=pydot.graph_from_dot_file('tree dot')
    name= 'tree'+ str(i_tree)
    graph.write_png(name+'.png')
    os.system('dot -Tpg tree.dot -o tree.png')
    i_tree+=1


# In[17]:





# In[1]:


#cross_validation 交差検定
#def get_score(clf,train_data, train_label):
#    train_data, test_data,train_label,test_label = cross_validation.train_test_split(train_data, train_label,test_size=0.2,random_state=0) #random_state=0
#    clf.fit(train_data, train_label)
#    print (clf.score(test_data, test_label) )   #cross_validation.train_test_splitは一定の割合が検証用データとなる
    
#def get_accuracy(clf,train_data, train_label):
#    scores = cross_validation.cross_val_score(clf,train_data, train_label, cv=10)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




#精度検証

#pred = clf.predict(test_data)
#fpr, tpr, thresholds = roc_curve(test_label, pred, pos_label=1)
#auc(fpr, tpr)
#accuracy_score(pred, test_label)



#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("decisiontree.pdf")
#Image(graph.create_png())
#from sklearn import tree
#for i,val in enumerate(clf.estimators_):
#   tree.export_graphviz(clf.estimators_[i], out_file='tree_%d.dot'%i)

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=17)

score_train_tmp = 0
score_test_tmp = 0

for train_index, test_index in kf.split(data):
    train_data, train_label= data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]

    # 構築データでモデル構築
    clf.fit( train_data, train_label)

    # 構築データの予測値
    pred_train = clf.predict(train_data)

    # 構築データのaccuracy
    auccuracy = accuracy_score(pred_train, train_label)

    #構築データのaccuracyを足していく
    score_train_tmp+=auccuracy

    #検証データの予測値
    pred_test = clf.predict(test_label)

    #検証データのaccuracy
    auccuracy = accuracy_score(pred_test, y_test)

    #検証データのaccuracyを足していく
    score_test_tmp+=auccuracy

#決定木の可視化
dot_data = StringIO()
i_tree=0

col=['weight','K length','K Perimeter length','K Radius','Total pixels','K area','3D K length','3D weight','3D K length','3D K Perimeter length','3D K Radius','Height in 3D','Average height in 3D','Luminance value']
for tree_in_forest in clf.estimators_:
    export_graphviz(tree_in_forest, out_file='tree dot',feature_names=col, max_depth=5)#max_depth=5
    (graph,)=pydot.graph_from_dot_file('tree dot')
    name= 'tree'+ str(i_tree)
    graph.write_png(name+'.png')
    os.system('dot -Tpg tree.dot -o tree.png')
    i_tree+=1


# In[ ]:




