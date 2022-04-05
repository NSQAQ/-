import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 目标函数
def h_fun(x,theta):
	X=x.copy()
	dem=1+np.exp(-X@theta)
	return 1/dem

# cost——function
def j_fun(x,y,theta):
	inner=np.multiply(y,np.log(h_fun(x,theta)))+np.multiply(1-y,np.log(h_fun(x,theta)))
	return -inner.sum()/len(x)

#梯度下降
def b_dg(x,y,theta,aphla):
	theta_num=theta.copy()
	times,j_num=[],[]
	for time in range(1000):
		for l in range(theta.shape[0]):
			theta_num[l][0]=theta_num[l][0]-aphla*(np.multiply(h_fun(x,theta)-y.T,x[:,l]).sum())
		times.append(time)
		j_num.append(j_fun(x,y,theta_num))
	plt.plot(times,j_num)
	plt.show()
	return theta_num
# 数据处理
data=pd.read_csv("./ex2data1.txt",names=["score_1","score_2","y"])
# 查看散点图
# data_y0=data[data.y==0]
# data_y1=data[data.y>0]
# plt.plot(data_y0.score_1,data_y0.score_2,"ro",data_y1.score_1,data_y1.score_2,"b*")
# plt.show()
data.insert(0,"x_0",np.ones(100))
x=np.matrix(data.iloc[:,0:3].values)
y=np.matrix(data.iloc[:,-1].values)
theta=np.matrix(np.zeros((3,1)),dtype=float)
#h_num=h_fun(x,theta)
#j_num=j_fun(x,y,theta)
theta=b_dg(x,y,theta,0.01)