import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
	dem = 1 + np.exp(-z)
	return 1 / dem


def cost_fun(theta, x, y, lambdas):
	frist = np.multiply(y, np.log(sigmoid(x @ theta)))
	second = np.multiply(1 - y, np.log(1 - sigmoid(x @ theta)))
	lambas_num = lambdas / (2 * x.shape[0]) * (np.power(theta, 2)).sum()
	return (frist + second).sum() / (-x.shape[0]) + lambas_num


# 特征映射，原有特征组合为多项式
def feature_mapping(x1, x2, power):
	data = {}  # 字典存储映射后的特征
	for i in np.arange(power + 1):
		for p in np.arange(i + 1):
			data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)  # 遍历进行特征多项式组合
	return pd.DataFrame(data)  # 字典转化为DataFrame并返回


def b_gd(theta, x, y, aphla, lambdas):
	theta_num = theta.copy()
	times = []
	cost_num = []
	for time in range(5000):
		sigmodl_num = sigmoid(x @ theta_num) - y
		for l in range(1, theta_num.shape[0]):
			theta_num[0, 0] = theta_num[0, 0] - aphla * np.multiply(sigmodl_num, x[:, 0]).sum() / x.shape[0]
			theta_num[l, 0] = theta_num[l, 0] - aphla * (
						np.multiply(sigmodl_num, x[:, l]).sum() / x.shape[0] + lambdas / x.shape[0] * theta_num[l, 0])
		# theta_num[l, 0] = theta_num[l, 0] - aphla * np.multiply(sigmodl_num, x[:, l]).sum()/x.shape[0]
		times.append(time)
		cost_num.append(cost_fun(theta_num, x, y, lambdas))
	plt.plot(times, cost_num)
	plt.show()
	return theta_num


# 特征映射：

# 获得数据并构造多项式
data = pd.read_csv("./ex2data2.txt", names=["x_1", "x_2", "y"])
x_1 = data["x_1"]
x_2 = data["x_2"]
x_data = feature_mapping(x_1, x_2, 6)
leng = data.shape[1]
x = np.matrix(x_data.values)
y = np.matrix(data.iloc[:, -1:leng])
theta = np.matrix(np.zeros((x_data.shape[1], 1)))
lambdas = 1
print(cost_fun(theta, x, y, lambdas))
theta = b_gd(theta, x, y, 0.1, lambdas)
print(cost_fun(theta, x, y, lambdas))
# 作图部分
data_y0 = data[data.y == 0]
data_y1 = data[data.y == 1]
plt.plot(data_y0.x_1.values, data_y0.x_2.values, "r*", data_y1.x_1.values, data_y1.x_2.values, "go")
# 作点
x0 = np.linspace(-1, 1, 500)
# 构造网格
xx, yy = np.meshgrid(x0, x0)
z = feature_mapping(xx.ravel(), yy.ravel(), 6).values@theta
z=z.reshape(xx.shape)
plt.contour(xx, yy, z, 0)
plt.show()
