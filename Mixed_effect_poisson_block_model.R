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

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
d <- 150   #�A�C�e����
k <- 7   #���ݕϐ���
N <- d*(d-1)/2   #���T���v����
vec <- rep(1, k)

##ID�Ɛ��ݕϐ��̐ݒ�
#���ݕϐ��𐶐�
Z <- rmnom(d, 1, runif(k, 1, 3))
z <- as.numeric(Z %*% 1:k)

#ID��ݒ�
id1 <- id2 <- c()
for(i in 1:(d-1)){
  id1 <- c(id1, rep(i, length((i+1):d)))
  id2 <- c(id2, (i+1):d)
}

##�����ϐ��̐���
#�p�����[�^�𐶐�
alpha <- rnorm(d, 0.8, 0.75)   #�ϗʌ��ʂ̃p�����[�^
theta <- matrix(rnorm(k*k, 0, 0.75), nrow=k, ncol=k)   #���ݕϐ��̃p�����[�^

#�|�A�\�����z�̕��ύ\��
mu <- alpha[id1] + alpha[id2] + (theta[z[id1], ] * Z[id2, ]) %*% vec
lambda <- exp(mu)

#�|�A�\�����z���牞���ϐ��𐶐�
y <- rpois(N, lambda)
sum(y); mean(y)
hist(y, xlab="�p�x", main="�A�C�e���Ԃ̏o���p�x", col="grey")
