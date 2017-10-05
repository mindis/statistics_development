
# coding: utf-8

# In[6]:

#####打ち切りデータのモデリング#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from scipy.special import gammaln
from scipy.misc import factorial
from pandas.tools.plotting import scatter_matrix
from numpy.random import *
from scipy import optimize
import seaborn as sns


# In[7]:

#####片側打ち切りモデル#####
####データの発生####
n = 10000   #サンプル数
p = 15   #説明変数数
b = uniform(-1.5, 4.5, p)   #回帰係数のパラメータ
b0 = 8.4   #切片
sigma = 8   #標準偏差
X = np.reshape(uniform(-1.0, 5.0, n*p), (n, p))   #説明変数


# In[8]:

##真のデータを発生
D = np.round(b0 + np.dot(X, b) + normal(0, sigma, n), 0)   #真の需要関数
S = np.round(b0 + np.dot(X, b) + uniform(0, 2.5, n), 0)   #真の供給関数


# In[21]:

#購買データを発生(需要が供給を上回っている場合供給を購買データとする)
B = np.zeros((n))
for i in range(n):
    if D[i] > S[i]:
        B[i] = S[i]
    else:
        B[i] = D[i]

Data0 = np.concatenate((B[:, np.newaxis], D[:, np.newaxis], S[:, np.newaxis]), axis=1)   #データを結合
Data = pd.DataFrame(Data0, columns=["B", "D", "S"])


# In[64]:

##打ち切りデータの指示変数を作成
z1 = np.array(Data.index[Data.D < Data.S]).astype(int)   #需要が満たされているデータ
z2 = np.array(Data.index[Data.D > Data.S]).astype(int)   #需要が供給を上回っているデータ


# In[129]:

####打ち切りデータモデルを推定####
##対数尤度の定義
beta = np.concatenate((np.reshape(np.array(b0), (1, )), b), axis=0)
beta0 = beta[0]
beta1 = beta[range(1, beta.shape[0])]

Mu = beta0 + np.dot(X, beta1)

#非打ち切りデータの対数尤度
var = pow(sigma, 2)
L1 = np.sum(-np.log(var) - pow((D - Mu)[z1], 2) / var)

#打ち切りデータの対数尤度

