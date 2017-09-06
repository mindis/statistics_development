#####多変量回帰モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from numpy.random import *
from scipy import optimize

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


##任意の分散共分散行列を発生させる
#パラメータを設定
col = 10
lower = -0.9
upper = 0.9
eigen_lower = 0
eigen_upper = 0.1
sigma_lower = 1.5
sigma_upper = 2.0

#相関行列を発生させる
Cor = CorM(col=col, lower=lower, upper=upper, eigen_lower=eigen_lower, eigen_upper=eigen_upper)
print(scipy.linalg.eigh(Cor)[0])   #正定値かどうか確認
print(np.round(Cor, 3))   #相関行列を確認

#分散共分散行列に変換
Cov = covmatrix(Cor=Cor, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
print(scipy.linalg.eigh(Cov)[0])
print(np.round(Cov, 3))


##多変量正規分布から乱数を発生させる
X = np.zeros((1000, col))
mu = np.zeros(col)

for i in range(X.shape[0]):
    X[i, ] = np.random.multivariate_normal(mu, Cor)

print(np.round(np.corrcoef(X.transpose()), 3))   #発生させた相関行列
print(np.round(Cor, 3))   #真の相関行列
