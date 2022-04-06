import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt


# 目标函数
def h_fun(z):
	return 1 / (1 + np.exp(-z))


# cost——function
def j_fun(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(h_fun(X *theta.T)))
	second = np.multiply((1 - y), np.log(1 - h_fun(X *theta.T)))
	return np.sum(first - second) / (len(X))


# 梯度下降
def b_dg(theta,X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)

	error = h_fun(X * theta.T) - y

	for i in range(parameters):
		term = np.multiply(error, X[:, i])
		grad[i] = np.sum(term) / len(X)

	return grad


# 数据处理
data = pd.read_csv("./ex2data1.txt", names=["score_1", "score_2", "y"])
# 查看散点图
# data_y0=data[data.y==0]
# data_y1=data[data.y>0]
# plt.plot(data_y0.score_1,data_y0.score_2,"ro",data_y1.score_1,data_y1.score_2,"b*")
# plt.show()
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
# theta = np.matrix(np.zeros((3, 1)), dtype=float)
# h_num=h_fun(x,theta)
# j_num=j_fun(x,y,theta)
print(j_fun(theta,X,y))
theta = opt.fmin_tnc(func=j_fun, x0=theta, fprime=b_dg, args=(X, y))[0]
print(j_fun(theta,X,y))
