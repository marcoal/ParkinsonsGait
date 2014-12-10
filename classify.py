from featureGen import FeatureGen
from featureSelect import *
from math import ceil
from matplotlib import pyplot
import numpy as np
from numpy.linalg import inv
import random
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score
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

def getTrainTestError(clf, X, Y):
    i = int(len(X * 0.9))
    trainX, testX = X[:i], X[i:]
    trainY, testY = Y[:i], Y[i:]
    clf.fit(trainX, trainY)
    
    # Training error
    output = clf.predict(trainX)
    numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, trainY)])
    numTotal = float(len(trainY))
    trainError = numWrong/numTotal

    # Test error
    output = clf.predict(testX)
    numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, testY)])
    numTotal = float(len(testY))
    testError = numWrong/numTotal

    print "Training error: {}, Test error: {}".format(trainError, testError)

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

def get_precision_recall(clf, X, Y):
    precisions = []
    recalls = []
    for i in range(10):
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1, random_state=0)
        clf.fit(trainX, trainY)

        # Test precision and recall
        output = clf.predict(testX)
        precisions.append(precision_score(testY, output))
        recalls.append(recall_score(testY, output))

    avgPrecision = sum(precisions)/len(precisions)
    avgRecall = sum(recalls)/len(recalls)

    print "Precision for {}: {}".format(clf.__class__.__name__, avgPrecision)
    print "Recall for {}: {}".format(clf.__class__.__name__, avgRecall)

def multiclass_AUC(clf, X, Y):
    # Binarize the output
    X, Y = np.array(X), np.array(Y)
    Y = label_binarize(Y, classes=list(set(Y)))
    n_classes = Y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                        random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    Y_score = classifier.fit(X_train, Y_train).predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print "AUC for multiclass {}: {}".format(clf.__class__.__name__, roc_auc["micro"])

def main():
    f = FeatureGen()
    # Binary classification
    X, Y = f.getXY(classifier='severity')
    Y = np.asarray(Y)
    
    nfeatures = len(X[0])
    max = 0
    best_sub = forward_search(nfeatures, tst_multiclass_AUC, [DecisionTreeClassifier(), X, Y])
    print best_sub

    X = np.array(X)
    X = X[:, best_sub]
    multiclass_AUC(DecisionTreeClassifier(), X, Y)
    cross_validate_accuracy(DecisionTreeClassifier(), X, Y)
    get_precision_recall(DecisionTreeClassifier(), X, Y)

if __name__ == "__main__":
    main()
