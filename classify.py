from featureGen import FeatureGen
import numpy as np
from numpy.linalg import inv
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

f = FeatureGen()
X, Y = f.getXY()

# Random Forest Classifier
# TODO: Training accuracy 99%, test 57%...
# Looks like we have a variance problem
clf = RandomForestClassifier()
clf.fit(X, Y)
output = clf.predict(X)
numCorrect = sum([int(predicted == actual) for predicted, actual in zip(output, Y)])
numTotal = float(len(Y))
print "Training accuracy for Random Forest: {}".format(numCorrect/numTotal)

scores = cross_val_score(clf, X, Y, cv=5)
print "Test accuracy for Random Forest: {}".format(sum(scores)/len(scores))

# Simple Unweighted Linear Regression
# TODO: it always predicts positive...
# Also can we use regression for a classification problem?
X, Y = np.matrix(X), np.matrix(Y)
theta = inv(X.T * X)*X.T*Y.T
output = [1 for pt in X]
numCorrect = np.sum((output == Y))
numTotal = float(Y.shape[1])
print "Accuracy for linear regression: {}".format(numCorrect / numTotal)