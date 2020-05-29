import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#「豚のトレーニングデータ」
dh = pd.read_csv("Traindatafor05072019.csv",header=None)


#体重を限定するとき
#a = df[df.iloc[:,16] <120]

#b = a[a.iloc[:,16] >= 80]


#c = b[b.iloc[:,13] >= 15]


# 説明変数、目的変数
X = dh.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]#.values
y = dh.iloc[:,[16]]#.values

#test data
do= pd.read_csv("070719data test.csv",header=None)

aa = do[do.iloc[:,16] <120]

bb = aa[aa.iloc[:,16] >= 80]

XX = bb.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]#.values
yy = bb.iloc[:,[16]]#.values,7,8,9,10,11,12,13,14,15


model=LinearRegression()
model.fit(X,y)


# 評価する  
predict = model.predict(XX)

rate_sum = 0

df1=pd.DataFrame(predict)
df1.to_csv('liner.csv',header=False,index=False)

rms = np.sqrt(mean_squared_error(predict,yy))
print(rms)


print(model.score(X,y ))# 決定係数(学習用)  　　　0.8815585422163619
print(model.score(XX, yy))  # 決定係数（テスト用）    0.8793248429755645

print(model.coef_)
print(model.intercept_)


# In[8]:


AK=bb.iloc[:,[0]].as_matrix()
BK=bb.iloc[:,[1]].as_matrix()
CK=bb.iloc[:,[2]].as_matrix()
DK=bb.iloc[:,[3]].as_matrix()
EK=bb.iloc[:,[4]].as_matrix()
FK=bb.iloc[:,[5]].as_matrix()
GK=bb.iloc[:,[6]].as_matrix()
HK=bb.iloc[:,[7]].as_matrix()
IK=bb.iloc[:,[8]].as_matrix()
JK=bb.iloc[:,[9]].as_matrix()
KK=bb.iloc[:,[10]].as_matrix()
LK=bb.iloc[:,[11]].as_matrix()
MK=bb.iloc[:,[12]].as_matrix()
NK=bb.iloc[:,[13]].as_matrix()
OK=bb.iloc[:,[14]].as_matrix()
PK=bb.iloc[:,[15]].as_matrix()
QK=bb.iloc[:,[16]].as_matrix()






for j in (0,(int)((len(QK)))): #len(EK)はデータの数　どこでもいい　エクセルの縦の列
   # weight1=(-2.67533123e-02*AK)+(2.97722839e-02*BK)+(-4.04737427e-02*CK)+(1.49796407e-01*DK)+(2.32328880e-04*EK)+(1.02673086e-04*FK)+(6.80263861e-02 *GK)+(-2.19853879e-02*HK)+(-1.81497145e-02*IK)+(2.05945970e-02*JK)+(2.13199508e-01*KK)+(1.06709573e-01*LK)+(-2.33964489e-02*MK)-45.73831583
    weight1=( 3.68368201e-03*AK)+(6.63189678e-03*BK)+(1.24047698e-02*CK)+(8.31096720e-02*DK)+(1.45897567e-04*EK)+(3.30985630e-04*FK)+(-5.39198684e-01*GK)+(7.18769396e+03*HK)+(-6.18754654e+00*IK)+(-1.62548522e-03*JK)+(-3.16207137e-02*KK)+(5.53030639e-02*LK)+(2.16089066e-01*MK)+(5.32116583e-02*NK)+(-6.17071139e-04*OK)+(2.16804212e-02*PK)-21.95898856
    
model=LinearRegression()
model.fit(weight1,QK)

Yprediction=model.predict(QK)
plt.figure('Weight')
plt.scatter(weight1,QK,color='green')  #点

plt.plot(QK,Yprediction,color='blue')
plt.ylabel('Actual')
plt.xlabel('Pre')
plt.show()



