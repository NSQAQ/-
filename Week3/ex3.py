import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
	dem=1+np.exp(-z)
	return 1/dem
def cost_fun(thate,x,y,learningRate):
	first=np.multiply(y,np.log(sigmoid(thate@x)))
	second=np.multiply(1-y,np.log(1-sigmoid(thate@x)))
	left=(first+second).sum()/-(x.shape[1])
	right= learningRate * (np.power(thate[:, 1:], 2).sum()) / (2 * x.shape[1])
	return left+right
# 反向传播
def bp(theta1,theta2,x,y):
	D_3,D_2=0,0
	a_2 = np.vstack((np.matrix(np.ones((1, 5000))), sigmoid(thate1 @ x)))
	a_3 = sigmoid(thate2 @ a_2)
	d_3=a_3-y
	d_2=np.multiply(theta2.T*d_3,np.multiply(a_3,1-a_3))
	D_3=D_3+d_3@a_3.T
	D_2=D_2+d_2@a_2.T

# 数据初始化
data_x=pd.read_excel("./data1x.xlsx",dtype=float,header=0)
data_x.insert(0,"x_0",1)
data_y=pd.read_excel("./data1y.xlsx",dtype=float)
x=np.matrix(data_x.iloc[:,:].values).T
y=np.matrix(data_y.iloc[:,:].values)
y_mat=np.matrix(np.zeros((10,5000)))
for l in range(5000):
	y_mat[int(y[l,0]-1),l]=1
thate1=np.matrix(np.random.rand(40,401))
thate2=np.matrix(np.random.rand(10,41))
z_2=sigmoid(thate1 @ x)
a_2 = np.vstack((np.matrix(np.ones((1, 5000))),z_2 ))

a_3 = sigmoid(thate2 @ a_2)
# print(cost_fun(thate1,x,z_2,1))
print(cost_fun(thate2,a_2,y_mat,1))
