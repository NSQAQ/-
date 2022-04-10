import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 激活函数
def sigmoid(z):
	dem=1+np.exp(-z)
	return 1/dem

# cost——function
def cost_fun(x,y,theta):
	frist=np.multiply(y, np.log(sigmoid(x @ theta)))
	second=np.multiply((1-y), np.log(1 - sigmoid(x @ theta)))
	return -np.sum(frist+second)/len(x)

#梯度下降
def b_dg(x,y,theta,alpha):
	theta_num=theta.copy()
	for time in range(100000):
		logistic_num= sigmoid(x @ theta_num)
		for l in range(x.shape[1]):
			theta_num[l,0]=theta_num[l,0]-alpha*(np.multiply((logistic_num-y),x[:,l])).sum()/x.shape[0]
	return theta_num
# 数据处理
data=pd.read_csv("./ex2data1.txt",names=["score_1","score_2","y"])
data.insert(0,"x_0",np.ones(100))
x=np.matrix(data.iloc[:,0:3].values)
y=np.matrix(data.iloc[:,-1].values).reshape((100,1))
theta=np.matrix(np.zeros((3,1)),dtype=float)
#h_num=h_fun(x@theta)
#j_num=cost_fun(x,y,theta)
print(cost_fun(x,y,theta))
theta=b_dg(x,y,theta,0.001)
print(cost_fun(x, y, theta))
# 查看散点图
data_y0=data[data.y==0]
data_y1=data[data.y>0]
x_0=range(100)
theta_list=theta.tolist()
y_0=[(-theta_list[0][0]-theta_list[1][0]*i)/(theta_list[2][0]) for i in x_0]
plt.plot(data_y0.score_1,data_y0.score_2,"ro",data_y1.score_1,data_y1.score_2,"b*",x_0,y_0,"-")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()