#####確率的潜在意味解析(トピックモデル)#####
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


####データを発生####
#データの設定
k = 8   #トピック数
d = 2000   #文書数
v = 300   #語彙数
w = poisson(250, d)   #1文書あたりの単語数



#ディレクリ分布のパラメータの設定
alpha0 = np.repeat(0.3, k)   #文書のディレリ事前分布のパラメータ
alpha1 = np.repeat(0.25, v)   #単語のディレクリ事前分布のパラメータ

#ディレクリ分布からパラメータを発生
theta = np.random.dirichlet(alpha0, d)   #文書のトピック分布をディレクリ乱数から発生
phi = np.random.dirichlet(alpha1, v)    #単語のトピック分布をデレクレリ乱数から発生



##多項分布から文書データを発生
WX = np.zeros((d, v), dtype='int')
Z = [i for i in range(d)]
vec = np.arange(1, k+1)

for i in range(d):
    
    #文書のトピックを生成
    z = multinomial(1, theta[i, :], w[i])   #文書のトピック分布を発生
    index_z = np.dot(z, vec)

    #トピック割当から単語を生成
    word = np.zeros((w[i], v))
    for j in range(w[i]):
        word[j, :] = multinomial(1, phi[index_z[j], :], 1)

    WX[i, :] = np.sum(word, axis=0)
    Z[i] = index_z



####トピックモデルのためのデータと関数の準備####
##それぞれの文書中の単語の出現をベクトルに並べる
##データ推定用IDを作成
ID_list = [i for i in range(d)]
wd_list = [i for i in range(d)]

#文書ごとに求人IDおよび単語IDを作成
for i in range(d):
    ID_list[i] = np.repeat(i, w[i])
    num1 = (WX[i, :] > 0) * np.arange(1, v+1)
    num2 = num1[num1!=0]
    W1 = WX[i, WX[i, :] > 0]
    number = np.repeat(num2, W1)
    wd_list[i] = number

