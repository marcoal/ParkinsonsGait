from featureGen import FeatureGen
from matplotlib import pyplot
import numpy as np
from numpy.linalg import inv
import random
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC

# Analyze mean force for each sensor
def analyzeSensorMeans(X, Y):
	print np.mean(X, axis=0)

# Analyze mean force for PD and nonPD subjects
def analyzeGlobalMeans(X, Y):
	a = zip(X, Y)
	nonPD = [x[1] for x, y in a if y == 0]
	PD = [x[1] for x, y in a if y == 1]
	nonPDmean, PDmean = np.mean(nonPD), np.mean(PD) 
	nonPDvar, PDvar = np.var(nonPD), np.var(PD)
	print "Mean: {}, variance: {}".format(nonPDmean, nonPDvar)
	print "Mean: {}, variance: {}".format(PDmean, PDvar)

def downsample(X, Y):
	numPositive = sum(Y)
	numNegative = len(Y) - numPositive
	numRemoved, i = 0, 0
	while numRemoved <= numPositive - numNegative:
		if Y[i] == 1:
			X.pop(i)
			Y.pop(i)
			numRemoved += 1
		i = random.randint(0, len(Y)-1)
	return X, Y

# Run Logistic Regression and plot train and test errors
def plotTrainTest(clf, X, Y):	
	trainingSizes = range(100, 250, 25)
	trainErrors, testErrors = [], []
	for i in trainingSizes:
		trainX, testX = X[:i], X[i:]
		trainY, testY = Y[:i], Y[i:]
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
		print sum(output), numTotal, float(sum(output))/numTotal
		testErrors.append(numWrong/numTotal)

	pyplot.plot(trainingSizes, trainErrors, label="Training error")
	pyplot.plot(trainingSizes, testErrors, label="Test error")
	pyplot.legend()
	pyplot.xlabel('Training set size')
	pyplot.ylabel('Error')
	pyplot.title('{} train and test error vs. training set size'.format(clf.__class__.__name__))
	pyplot.show()

def crossValidate(clf, X, Y):
	scores = cross_val_score(clf, X, Y, cv=5)
	print "Test accuracy for {}: {}".format(clf.__class__.__name__, sum(scores)/len(scores))

def main():
	# Create feature and label vectors
	f = FeatureGen()
	X, Y = f.getXY()
	# X, Y = downsample(X, Y)
	plotTrainTest(LogisticRegression(), X, Y)
	crossValidate(LogisticRegression(), X, Y)

if __name__ == "__main__":
	main()
