import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report  # 这个包是评价报告


def load_weight(path):
	data = sio.loadmat(path)
	return data['Theta1'], data['Theta2']


def load_data(path):
	data = sio.loadmat(path)
	X = data.get('X')
	y = data.get('y')
	y = y.reshape(y.shape[0])
	return X, y
def sigmoid(z):
	return 1 / (1 + np.exp(-z))


theta1, theta2 = load_weight('ex3weights.mat')
X, y = load_data('ex3data1.mat')

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
a1 = X
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
# argmax是返回最大值的索引，因为索引是从0开始所以要+1
y_pred = np.argmax(a3, axis=1)
print(classification_report(y, y_pred))