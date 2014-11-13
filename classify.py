from featureGen import FeatureGen
from featureSelect import forward_search
from featureSelect import cv_accuracy
from matplotlib import pyplot
import numpy as np
from numpy.linalg import inv
import random
import sklearn
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

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

# Run Logistic Regression and plot train and test error
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
		testErrors.append(numWrong/numTotal)
    
	pyplot.plot(trainingSizes, trainErrors, label="Training error")
	pyplot.plot(trainingSizes, testErrors, label="Test error")
	pyplot.legend()
	pyplot.xlabel('Training set size')
	pyplot.ylabel('Error')
	pyplot.title('{} train and test error vs. training set size'.format(clf.__class__.__name__))
	pyplot.show()

def cross_validate(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='roc_auc')
    avgAuc = sum(scores)/float(len(scores))
    print "Cross val score for {}: {}".format(clf.__class__.__name__, avgAuc)
    return avgAuc

def main():
    # Create feature and label vectors
    f = FeatureGen()
    X, Y = f.getXY()
    plotTrainTest(LogisticRegression(class_weight='auto'), X, Y)
    cross_validate(LogisticRegression(class_weight='auto'), X, Y)

if __name__ == "__main__":
    main()
