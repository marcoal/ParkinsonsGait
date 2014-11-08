from featureGen import FeatureGen
from matplotlib import pyplot
import numpy as np
from numpy.linalg import inv
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

# Create feature and label vectors
f = FeatureGen()
X, Y = f.getXY()

# Analyze mean force of PD and non-PD subjects
# a = zip(X, Y)
# nonPD = [x[1] for x, y in a if y == 0]
# PD = [x[1] for x, y in a if y == 1]
# nonPDmean, PDmean = np.mean(nonPD), np.mean(PD) 
# nonPDvar, PDvar = np.var(nonPD), np.var(PD)
# print "Mean: {}, variance: {}".format(nonPDmean, nonPDvar)
# print "Mean: {}, variance: {}".format(PDmean, PDvar)

# Run Logistic Regression and plot train and test errors
trainingSizes = range(100, 250, 25)
trainErrors, testErrors = [], []
for i in trainingSizes:
	trainX, testX = X[:i], X[i:]
	trainY, testY = Y[:i], Y[i:]
	clf = LogisticRegression()
	clf.fit(trainX, trainY)
	
	# Training error
	output = clf.predict(trainX)
	numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, trainY)])
	numTotal = float(len(trainY))
	trainErrors.append(numWrong/numTotal)

	# Test error
	output = clf.predict(testX)
	numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, testY)])
	numTotal = float(len(testY))
	testErrors.append(numWrong/numTotal)

pyplot.plot(trainingSizes, trainErrors, label="Training error")
pyplot.plot(trainingSizes, testErrors, label="Test error")
pyplot.legend()
pyplot.xlabel('Training set size')
pyplot.ylabel('Error')
pyplot.title('Logistic regression train and test error vs. training set size')
pyplot.show()


# scores = cross_val_score(clf, X, Y, cv=5)
# print "Test accuracy for Random Forest: {}".format(sum(scores)/len(scores))

# # Simple Unweighted Linear Regression
# # TODO: it always predicts positive...
# # Also can we use regression for a classification problem?
# X, Y = np.matrix(X), np.matrix(Y)
# theta = inv(X.T * X)*X.T*Y.T
# output = [1 for pt in X]
# numCorrect = np.sum((output == Y))
# numTotal = float(Y.shape[1])
# print "Accuracy for linear regression: {}".format(numCorrect / numTotal)