from sklearn.datasets import load_iris
from sklearn import tree
#import a dataset
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

#Once trained, you can plot the tree with the plot_tree function:
tree.plot_tree(clf.fit(iris.data, iris.target)) 