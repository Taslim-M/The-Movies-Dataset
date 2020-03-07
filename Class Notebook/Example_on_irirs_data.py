import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. 
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear')
svc.fit(X, y)

print (svc.score (X, y)); # print the trainig score(accuracy)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

# gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
# Higher the value of gamma, will try to exact fit the as per training data set 
# i.e. generalization error and cause over-fitting problem.

#C: Penalty parameter C of the error term. 
#It also controls the trade off between smooth decision boundary and 
# classifying the training points correctly.

# Nonlinear kernel
svc = svm.SVC(kernel='rbf', C=1,gamma=1).fit(X, y)
# try to increase gamma from 1 to 10 to 100

# try to increase C from 1 to 100 to 1000
