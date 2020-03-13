import numpy as np
from sklearn.model_selection import train_test_split

### Create a dataset of features as 2D points and labels
features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels = np.array([1, 1, 1, 2, 2, 2])

### Split arrays or matrices into random train and test subsets
features_subset1, features_subset2, labels_subset1, labels_subset2 = train_test_split(features, labels, test_size=0.50)

### print the subsets, for visualization purposes
print('Features for subset1 are: ', features_subset1)
print('Features for subset2 are: ', features_subset2)

######################################################################

### import the sklearn module for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

### create the classifier
clf = DecisionTreeClassifier()

### fit (train) the classifier on the training features and labels
clf.fit(features_subset1, labels_subset1) 

### Predict the labels for features_part2 and print them
print('Predicion result on subset2 is: ', clf.predict(features_subset2))

### Find the prediction accuracy of the model 
print('Predicion accuracy is: ', clf.score(features_subset2, labels_subset2))

######################################################################

### import the sklearn module for NaiveBayes Classifier
from sklearn.naive_bayes import GaussianNB 

### create the classifier
clf = GaussianNB()

### fit (train) the classifier on the training features and labels
clf.fit(features_subset1, labels_subset1) 

### Predict the labels for features_part2 and print them
print('Predicion result on subset2 is: ', clf.predict(features_subset2))

### Find the prediction accuracy of the model 
print('Predicion accuracy is: ', clf.score(features_subset2, labels_subset2))

######################################################################

### import the sklearn module for Support Vector Classifier (SVC)
from sklearn.svm import SVC

### create the classifier
clf = SVC()

### fit (train) the classifier on the training features and labels
clf.fit(features_subset1, labels_subset1) 

### Predict the labels for features_part2 and print them
print('Predicion result on subset2 is: ', clf.predict(features_subset2))

### Find the prediction accuracy of the model 
print('Predicion accuracy is: ', clf.score(features_subset2, labels_subset2))
