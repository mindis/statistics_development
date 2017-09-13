
# coding: utf-8

# In[1]:

#####混合多項分布モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from numpy.random import *
from scipy import optimize
import seaborn as sns


# In[6]:

####データの発生####
##データの設定
N = 1000   #サンプル数
k = 25   #変数数
seg = 5   #セグメント数
g = poisson(30, N)   #購買数


# In[11]:

##パラメータの設定
np.reshape(uniform(1, 15, seg*k), (seg, k))


# In[ ]:



