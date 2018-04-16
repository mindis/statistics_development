#####mixed effect poisson block model#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(flexmix)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(506832)

####データの発生####
##データの設定
d <- 150   #アイテム数
k <- 7   #潜在変数数
N <- d*(d-1)/2   #総サンプル数
vec <- rep(1, k)

##IDと潜在変数の設定
#潜在変数を生成
Z <- rmnom(d, 1, runif(k, 1, 3))
z <- as.numeric(Z %*% 1:k)

#IDを設定
id1 <- id2 <- c()
for(i in 1:(d-1)){
  id1 <- c(id1, rep(i, length((i+1):d)))
  id2 <- c(id2, (i+1):d)
}

##応答変数の生成
#パラメータを生成
alpha <- rnorm(d, 0.8, 0.75)   #変量効果のパラメータ
theta <- matrix(rnorm(k*k, 0, 0.75), nrow=k, ncol=k)   #潜在変数のパラメータ

#ポアソン分布の平均構造
mu <- alpha[id1] + alpha[id2] + (theta[z[id1], ] * Z[id2, ]) %*% vec
lambda <- exp(mu)

#ポアソン分布から応答変数を生成
y <- rpois(N, lambda)
sum(y); mean(y)
hist(y, xlab="頻度", main="アイテム間の出現頻度", col="grey")

