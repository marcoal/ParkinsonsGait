from featureGen import FeatureGen
from featureSelect import *
from math import ceil
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

# Analyze mean force for each sensor
def analyzeFeatureMeans(X, Y):
    posX = [x for x,y in zip(X,Y) if y == 1]
    negX = [x for x,y in zip(X,Y) if y == 0]
    print 'PD means: {}'.format(np.mean(posX, axis=0))
    print 'Non-PD means: {}'.format(np.mean(negX, axis=0))


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

def getTrainTestAUC(clf, X, Y):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf.fit(trainX, trainY)

    # Training AUC
    output = clf.predict(trainX)
    trainingAUC = roc_auc_score(trainY, output)

    # Test AUC
    output = clf.predict(testX)
    testAUC = roc_auc_score(testY, output)

    print "Training AUC: {}, Test AUC: {}".format(trainingAUC, testAUC)
    return trainingAUC, testAUC 

def cross_validate_AUC(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='roc_auc')
    avgAuc = sum(scores)/float(len(scores))
    print "Cross val AUC score for {}: {}".format(clf.__class__.__name__, avgAuc)
    
def cross_validate_accuracy(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=10)
    avgAccuracy = sum(scores)/float(len(scores))
    print "Cross val accuracy score for {}: {}".format(clf.__class__.__name__, avgAccuracy)

def main():
    f = FeatureGen()
    # Binary classification
    X, Y = f.getXY(classifier='PD')
    Y = np.asarray(Y)
    #cross_validate_AUC(LogisticRegression(class_weight='auto'), X, Y)
    nfeatures = len(X[0])

    max = 0
    best_sub = []
    for i in range(1, nfeatures):
        feat_sub = forward_search(nfeatures, i, tst_auc, [LogisticRegression(class_weight='auto'), X, Y])
        current = tst_auc(feat_sub, LogisticRegression(class_weight='auto'), X, Y)
        if current > max:
            max = current
            best_sub = feat_sub

    print best_sub
    print "Best Accuracy:{} + , X, Y)".format(max)
    print f.getFeatures_names()[best_sub]

    #cross_validate_accuracy(LogisticRegression(class_weight='auto'), X, Y)

    # Severity classification
    #X, Y = f.getXY(classifier='severity')
    #cross_validate_accuracy(LogisticRegression(class_weight='auto'), X, Y)
    

if __name__ == "__main__":
    main()
