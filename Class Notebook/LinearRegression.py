import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3] ])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3


# Visualize your data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0],X[:,1], y)
plt.scatter(X[:,0],X[:,1], y, color='darkorange', label='data')
plt.show()

reg = LinearRegression()
plane = reg.fit(X, y)

print(reg.score(X, y)) # R squared
print(reg.coef_) # Coefficients
print(reg.intercept_ ) # y-intercept
print(reg.predict(np.array([[3, 5]])))


