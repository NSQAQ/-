import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
	dem=1+np.exp(-z)
	return 1/dem
def cost_fun(theta,x,y):
	pass
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
thate2=np.matrix(np.random.rand(10,40))
sigmoid_mu=sigmoid(thate1@x)
