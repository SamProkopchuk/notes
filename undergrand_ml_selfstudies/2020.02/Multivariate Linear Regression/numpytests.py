import numpy as np

data = np.genfromtxt('psytest.csv', delimiter=',')
data = np.delete(data, 0, axis=0)
y = data[:,3]
data = np.delete(data, 3, axis=1)
print(data)
print(data.shape)
print(y)

y = y.reshape(5,5)
print(y)
means = y.T.mean(axis=1)
print(means)
print(np.sum(y))
print(y.sum())