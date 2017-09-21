
# coding: utf-8

# In[67]:

#####混合多変量正規分布モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from pandas.tools.plotting import scatter_matrix
from numpy.random import *
from scipy import optimize
import seaborn as sns


# In[68]:

####任意の相関行列(分散共分散行列)を作成する関数####
##任意の相関行列を作る関数
def CorM(col, lower, upper, eigen_lower, eigen_upper):
    #相関行列の初期値を定義する
    cov_vec = (upper - lower) *rand(col*col) + lower   #相関係数の乱数ベクトルを作成
    rho = np.reshape(np.array(cov_vec), (col, col)) * np.tri(col)   #乱数ベクトルを下三角行列化
    Sigma = np.diag(np.diag(rho + rho.T) + 1) - (rho + rho.T)   #対角成分を1にする
    
    #相関行列を正定値行列に変更
    #固有値分解を実行
    eigen = scipy.linalg.eigh(Sigma)
    eigen_val = eigen[0] 
    eigen_vec = eigen[1]
    
    #固有値が負の数値を正にする
    for i in range(eigen_val.shape[0]-1):
        if eigen_val[i] < 0:
            eigen_val[i] = (eigen_upper - eigen_lower) * rand(1) + eigen_lower
            
    #新しい相関行列の定義と対角成分を1にする
    Sigma = np.dot(np.dot(eigen_vec, np.diag(eigen_val)), eigen_vec.T)
    normalization_factor = np.dot(pow(np.diag(Sigma), 0.5)[:, np.newaxis], pow(np.diag(Sigma), 0.5)[np.newaxis, :])
    Cor = Sigma / normalization_factor
    return Cor


##相関行列から分散共分散行列に変換する関数
def covmatrix(Cor, sigma_lower, sigma_upper):
    sigma = (sigma_upper - sigma_lower) * rand(np.diag(Cor).shape[0]) + sigma_lower
    sigma_factor = np.dot(sigma[:, np.newaxis], sigma[np.newaxis, :])
    Cov = Cor * sigma_factor
    return Cov


# In[69]:

####データの発生####
##データの設定
n = 1000   #セグメントのサンプル数
seg = 4   #セグメント数
N = n*seg   #総サンプル数
k = 5   #パラメータ数


# In[70]:

##セグメント割当の設定
seg_id = np.array([])
for i in range(seg):
    seg_id = np.append(seg_id, np.repeat(i+1, n))


# In[83]:

##多変量正規分布からセグメントごとにデータを発生させる
#パラメータの設定
#パラメータと応答変数の格納用配列
Cor0 = np.zeros((k, k, seg))
Cov0 = np.zeros((k, k, seg))
Mu0 = np.zeros((seg, k))
Y = np.zeros((N, k))

#セグメントごとにパラメータを設定して応答変数を発生させる
for i in range(seg):
    lower = uniform(-0.6, 0.25)
    upper = uniform(0.25, 0.75)
    Cor0[:, :, i] = CorM(col=k, lower=-0.55, upper=0.8, eigen_lower=0.01, eigen_upper=0.2)
    Cov0[:, :, i] = covmatrix(Cor=Cor0[:, :, i], sigma_lower=0.7, sigma_upper=1.75)
    Mu0[i, :] = uniform(-4, 4, k)
    Y[seg_id==i+1, :] = np.random.multivariate_normal(Mu0[i, :], Cov0[:, :, i], n)


# In[132]:

##発生させた変数の集計
#散布図行列をプロット
Y_pd = pd.DataFrame(np.concatenate((seg_id[:, np.newaxis], Y), axis=1))
scatter_matrix(Y_pd[[1, 2, 3, 4, 5]], diagonal='kde', color='k', alpha=0.3)
plt.show()

#基礎集計



# In[166]:

####EMアルゴリズムで混合多変量正規分布を推定####
##多変量正規分布の尤度関数を定義
def dmv(x, mu, Cov, k):
    er = x - mu
    Cov_inv = np.linalg.inv(Cov) 
    LLo = 1 / (np.sqrt(pow((2 * np.pi), k) * np.linalg.det(Cov))) * np.exp(np.dot(np.dot(-er, Cov_inv), er) / 2)
    return(LLo)


# In[191]:

##観測データの対数尤度と潜在変数zの定義
LLind = np.zeros((N, seg))

for s in range(seg):
    mean_vec = Mu0[s, :]
    cov = Cov0[:, :, s]

    for i in range(N):
        LLind[i, s] = dmv(Y[i, :], mean_vec, cov, k)


# In[193]:

pd.DataFrame(LLind)


# In[158]:

Cov_inv = np.linalg.inv(Cov0[:, :, 0])
er = Y[0, :] - Mu0[0, :] 


# In[162]:

np.exp(np.dot(np.dot(-er, Cov_inv), er))


# In[159]:

Cov_inv


# In[ ]:



