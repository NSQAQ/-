import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# J函数
def J(t, x, y):
	num_sum = ((t.T @ x - y[:, 0]) ** 2).sum()
	return num_sum / (2 * len(y))


# batch 梯度下降（）
def b_gd(theta, alpha, x, y):
	theta_num = theta
	times = []
	j_num = []
	for time in range(5000):
		theta_num[0][0] = theta[0][0] - alpha * (((theta.T @ x - y.T)*x[0]).sum()) / len(y)
		theta_num[1][0] = theta[1][0] - alpha * (((theta.T @ x - y.T) * x[1]).sum()) / len(y)
		times.append(time)
		j_num.append(J(theta, x, y))
	theta = theta_num
	# 绘制J函数图像，检查梯度下降是否收敛
	plt.plot(times, j_num, 'b')
	plt.show()


# return theta   # 因为python的函数参数为可变长参数时，是引用传值。因此无需返回theta


# 正规方程
def normal(x, y):
	X = x.T
	t = (np.linalg.inv(X.T @ X)) @ X.T @ y
	return t


#
# 导入数据
data = pd.read_csv('ex1data1.txt', sep=',', names=["population", "profit"])
# 绘制图像,查看数据集的分布情况
# plt.plot(data["population"], data["profit"],'*b')
# plt.show()
# 构建数据
x = np.ones((2, len(data)), dtype=float)  # 构建x
x[1] = np.array(data["population"].values)
y = np.array(data["profit"].values, dtype=float).reshape(97, 1)  # 构建y
theta = np.array([[0], [0]], dtype=float)  # 构建特征变量
alpha = 0.01  # 设置学习率
# 测试
b_gd(theta, alpha, x, y)
theta_N = normal(x, y)
