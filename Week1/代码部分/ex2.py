import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 正规方程
def Normal(x, y):
	theta = np.linalg.inv((x.T @ x)) @ x.T @ y
	return theta


# J函数
def cost_fun(t, X, y):
	h_fun = np.multiply((X @ t - y), (X @ t - y))
	j_fun = h_fun.sum() / (len(X) * 2)
	return j_fun


# 梯度下降
def b_gd(x,y,theta,alpha):
	theta_num=theta.copy()
	j_num=[]
	times=[]
	for time in range(10000):
		h_fun=x@theta_num
		for l in range(x.shape[1]):
			theta_num[l,0]=theta_num[l,0]-alpha*(np.multiply((h_fun-y),x[:,l])).sum()/x.shape[0]
		j_num.append(cost_fun(theta_num, x, y))
		times.append(time)
	theta=theta_num
	plt.plot(times,j_num,'b')
	plt.show()
	return theta

"""
特征缩放五种常用方法，num为选择方法的序号
1为最大最小值归一化（min-max normalization）
2为均值归一化（mean normalization）
3为标准化 / z值归一化（standardization / z-score normalization）
4为最大绝对值归一化（max abs normalization ）
5为稳键标准化（robust standardization）
"""
def num_fc(l, num):
	# 最大最小值归一化（min-max normalization）
	if num == 1:
		for m in range(1, l.shape[1]):
			l[:, m] = (l[:, m] - l[:, m].min()) / (l[:, m].max() - l[:, m].min())
		return l
	# 均值归一化（mean normalization）
	elif num == 2:
		for m in range(1, l.shape[1]):
			l[:, m] = (l[:, m] - l[:, m].mean()) / (l[:, m].max() - l[:, m].min())
		return l
	# 标准化 / z值归一化（standardization / z-score normalization）
	elif num == 3:
		for m in range(1, l.shape[1]):
			l[:, m] = (l[:, m] - l[:, m].mean()) / l[:, m].std()
		return l
	# 最大绝对值归一化（max abs normalization ）
	elif num == 4:
		for m in range(1, l.shape[1]):
			l[:, m] = l[:, m] / np.fabs(l[:, m].max())
		return l
	# 稳键标准化（robust standardization）
	else:
		for m in range(1, l.shape[1]):
			x_array = np.array(l[:, m])
			l[:, m] = (l[:, m] - np.median(x_array)) / (
						np.quantile(x_array, 0.75, interpolation="higher") - np.quantile(x_array, 0.25,
						                                                                 interpolation="lower"))
		return l


data = pd.read_csv("ex1data2.txt", sep=",", names=["area", "room_num", "profit"])
#data=(data-data.mean())/data.std()
data.insert(0, "x_0", np.ones(47))
data1=np.matrix(data.iloc[:,:].values)
data1= num_fc(data1, 2)
#data=np.matrix(data.iloc[:,:].values,dtype=float)
x = np.matrix(data1[:,:3])
y = np.matrix(data1[:,-1])
theta = np.matrix(np.zeros((3, 1)))
x= num_fc(x, 5)
theta_B= b_gd(x, y, theta, 0.01)
theta_N = Normal(x, y)
