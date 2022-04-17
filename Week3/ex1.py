import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path):
	data = sio.loadmat(path)
	X = data.get('X')
	y = data.get('y')
	y = y.reshape(y.shape[0])
	# 图像是镜像的，所以需要对20*20的简单矩阵进行转置
	if 1:
		X = np.array([im.reshape((20, 20)).T for im in X])
		X = np.array([im.reshape(400) for im in X])
	return X, y


def plot_to_image(image):
	fig, ax = plt.subplots(figsize=(1, 1))
	ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))


def plot_100_image(X):
	test_image = np.random.choice(np.arange(X.shape[0]), 100)
	test_image = X[test_image, :]
	fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
	for n in range(0, 10):
		for m in range(0, 10):
			ax_array[n, m].matshow(test_image[10 * n + m, :].reshape((20, 20)), cmap=matplotlib.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
	return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
	theta_j1_to_n = theta[1:]
	regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

	return cost(theta, X, y) + regularized_term


def gradient(theta, X, y):
	return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def regularized_gradient(theta, X, y, l=1):
	theta_j1_to_n = theta[1:]
	regularized_theta = (l / len(X)) * theta_j1_to_n

	regularized_term = np.concatenate([np.array([0]), regularized_theta])

	return gradient(theta, X, y) + regularized_term


def logistic_regression(X, y, l=1):
	theta = np.zeros(X.shape[1])

	# train it
	res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient,options={'disp': True})
	# get trained parameters
	final_theta = res.x

	return final_theta


def predict(x, theta):
	prob = sigmoid(x @ theta)
	return (prob >= 0.5).astype(int)


X, y = load_data('ex3data1.mat')
raw_y=y.copy()
# 显示手写图片
plot_100_image(X)
plt.show()
X = np.insert(X, 0, 1, axis=1)
y_matrix = []
for k in range(1, 11):
	y_matrix.append((y == k).astype(int))
print(y_matrix[-1])
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)
t0 = logistic_regression(X, y[0])
y_pred = predict(X, t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))
# 训练K维模型
k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
prob_matrix = sigmoid(X @ k_theta.T)
y_pred = np.argmax(prob_matrix, axis=1)
y_answer = raw_y.copy()
y_answer[y_answer==10] = 0
print(classification_report(y_answer, y_pred))